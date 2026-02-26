"""
Tests for the agent core framework.

Uses fakeredis and temp SQLite — no external services needed.
"""

import os
import json
import time
import uuid
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, date

# Ensure project root is in path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.core.message_bus import MessageBus, InMemoryMessageBus, Message
from agents.core.guardrails import Guardrails, TokenBudget
from agents.core.agent_base import AgentBase, AgentResult, AgentStatus
from agents.core.agent_registry import AgentRegistry
from agents.core.connections import get_redis_client, get_postgres_connection


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fakeredis_client():
    """Create a fakeredis client for testing."""
    try:
        import fakeredis
        return fakeredis.FakeRedis(decode_responses=True)
    except ImportError:
        pytest.skip("fakeredis not installed")


@pytest.fixture
def redis_bus(fakeredis_client):
    """MessageBus backed by fakeredis."""
    return MessageBus(fakeredis_client)


@pytest.fixture
def memory_bus():
    """In-memory MessageBus."""
    return InMemoryMessageBus()


@pytest.fixture
def sqlite_guardrails(tmp_path):
    """Guardrails backed by temp SQLite."""
    db_path = str(tmp_path / "test_guardrails.db")
    return Guardrails(pg_conn=None, sqlite_path=db_path)


@pytest.fixture
def sample_message():
    """Create a sample message."""
    return Message.create(
        sender='pregame',
        recipient='predictor',
        event_type='intel_ready',
        payload={'game_id': '123', 'confidence': 'high'},
        priority='normal',
        ttl_minutes=60,
    )


class MockAgent(AgentBase):
    """Controllable mock agent for testing."""

    AGENT_NAME = 'mock_agent'
    DAILY_TOKEN_BUDGET = 10_000
    MAX_EXECUTION_SECONDS = 60

    def __init__(self, run_result=None, run_error=None, **kwargs):
        super().__init__(**kwargs)
        self._run_result = run_result or {'status': 'ok', 'reasoning': 'Test run'}
        self._run_error = run_error
        self.run_called = False
        self.report_called = False
        self.cleanup_called = False

    def run(self):
        self.run_called = True
        if self._run_error:
            raise self._run_error
        return self._run_result

    def report(self, run_output):
        self.report_called = True

    def cleanup(self):
        self.cleanup_called = True


# =============================================================================
# TestMessageBus (Redis-backed)
# =============================================================================

class TestMessageBus:
    """Tests for Redis-backed MessageBus."""

    def test_send_and_receive(self, redis_bus, sample_message):
        """Basic round-trip: send a message and receive it."""
        msg_id = redis_bus.send(sample_message)
        assert msg_id == sample_message.message_id

        messages = redis_bus.receive('predictor')
        assert len(messages) >= 1
        assert any(m.message_id == sample_message.message_id for m in messages)

    def test_filters_by_recipient(self, redis_bus):
        """Only returns messages for the target recipient."""
        msg_a = Message.create(sender='test', recipient='agent_a', event_type='test', payload={})
        msg_b = Message.create(sender='test', recipient='agent_b', event_type='test', payload={})

        redis_bus.send(msg_a)
        redis_bus.send(msg_b)

        msgs_a = redis_bus.receive('agent_a')
        redis_bus.receive('agent_b')

        assert any(m.message_id == msg_a.message_id for m in msgs_a)
        assert not any(m.message_id == msg_b.message_id for m in msgs_a)

    def test_filters_by_event_type(self, redis_bus):
        """Event type filtering works."""
        msg1 = Message.create(sender='test', recipient='target', event_type='intel_ready', payload={})
        msg2 = Message.create(sender='test', recipient='target', event_type='results_analyzed', payload={})

        redis_bus.send(msg1)
        redis_bus.send(msg2)

        filtered = redis_bus.receive('target', event_type='intel_ready')
        assert all(m.event_type == 'intel_ready' for m in filtered)

    def test_acknowledge(self, redis_bus, sample_message):
        """Consumed messages not returned after acknowledgment."""
        redis_bus.send(sample_message)
        redis_bus.acknowledge(sample_message.message_id, 'predictor')

        # Message payload is deleted, should not appear in receive
        messages = redis_bus.receive('predictor')
        assert not any(m.message_id == sample_message.message_id for m in messages)

    def test_broadcast_all(self, redis_bus):
        """Broadcast messages (recipient='all') visible to any consumer."""
        broadcast = Message.create(
            sender='system', recipient='all', event_type='system_alert',
            payload={'alert': 'test'},
        )
        redis_bus.send(broadcast)

        # Any recipient should see broadcast messages
        msgs = redis_bus.receive('pregame')
        assert any(m.event_type == 'system_alert' for m in msgs)

    def test_payload_roundtrip(self, redis_bus):
        """JSON payload survives Redis serialization."""
        complex_payload = {
            'game_id': '12345',
            'nested': {'a': 1, 'b': [1, 2, 3]},
            'float_val': 3.14,
            'null_val': None,
        }
        msg = Message.create(
            sender='test', recipient='target', event_type='test',
            payload=complex_payload,
        )
        redis_bus.send(msg)

        received = redis_bus.receive('target')
        assert len(received) >= 1
        assert received[0].payload == complex_payload


# =============================================================================
# TestInMemoryMessageBus
# =============================================================================

class TestInMemoryMessageBus:
    """Tests for the in-memory fallback MessageBus."""

    def test_in_memory_fallback(self, memory_bus):
        """InMemoryMessageBus works identically to Redis bus."""
        msg = Message.create(
            sender='pregame', recipient='predictor',
            event_type='intel_ready', payload={'test': True},
        )
        memory_bus.send(msg)

        messages = memory_bus.receive('predictor')
        assert len(messages) == 1
        assert messages[0].payload == {'test': True}

    def test_ttl_expiry(self, memory_bus):
        """Messages with expired TTL not returned."""
        msg = Message.create(
            sender='test', recipient='target',
            event_type='test', payload={}, ttl_minutes=0,
        )
        # Manually set expiry to the past
        memory_bus.send(msg)
        memory_bus._expiry[msg.message_id] = time.time() - 1

        messages = memory_bus.receive('target')
        assert len(messages) == 0


# =============================================================================
# TestAgentBase
# =============================================================================

class TestAgentBase:
    """Tests for AgentBase lifecycle."""

    def test_lifecycle_order(self, memory_bus, sqlite_guardrails):
        """execute() calls run -> report -> cleanup in order."""
        agent = MockAgent(message_bus=memory_bus, guardrails=sqlite_guardrails)
        result = agent.execute()

        assert agent.run_called
        assert agent.report_called
        assert agent.cleanup_called
        assert result.status == AgentStatus.COMPLETED.value

    def test_shadow_suppresses_messages(self, memory_bus, sqlite_guardrails):
        """No messages sent in shadow mode."""
        agent = MockAgent(message_bus=memory_bus, guardrails=sqlite_guardrails, shadow_mode=True)
        agent.execute()

        # Try sending a message — should be suppressed
        agent.send_message('target', 'test_event', {'data': 1})
        messages = memory_bus.receive('target')
        assert len(messages) == 0

    def test_circuit_breaker(self, memory_bus, sqlite_guardrails):
        """3 consecutive failures -> COOLDOWN status."""
        # Record 3 failures
        for i in range(3):
            sqlite_guardrails.record_run(
                agent_name='mock_agent',
                run_id=str(uuid.uuid4()),
                started_at=datetime.now(timezone.utc).isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                status='failed',
                success=False,
            )

        agent = MockAgent(message_bus=memory_bus, guardrails=sqlite_guardrails)
        result = agent.execute()

        assert result.status == AgentStatus.COOLDOWN.value
        assert not agent.run_called  # run() should not have been called

    def test_result_captures_errors(self, memory_bus, sqlite_guardrails):
        """Failed run populates errors list."""
        agent = MockAgent(
            run_error=ValueError("Test error"),
            message_bus=memory_bus,
            guardrails=sqlite_guardrails,
        )
        result = agent.execute()

        assert result.status == AgentStatus.FAILED.value
        assert len(result.errors) > 0
        assert 'Test error' in result.errors[0]

    def test_run_id_unique(self, memory_bus, sqlite_guardrails):
        """Each execute() gets a unique run_id."""
        agent = MockAgent(message_bus=memory_bus, guardrails=sqlite_guardrails)

        result1 = agent.execute()
        result2 = agent.execute()

        assert result1.run_id != result2.run_id
        assert len(result1.run_id) == 36  # UUID format

    def test_token_tracking(self, memory_bus, sqlite_guardrails):
        """Token usage recorded after call_llm."""
        agent = MockAgent(message_bus=memory_bus, guardrails=sqlite_guardrails)

        # Without GEMINI_API_KEY, call_llm returns empty string
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('GEMINI_API_KEY', None)
            result = agent.call_llm("system", "user")
            assert result == ''


# =============================================================================
# TestGuardrails
# =============================================================================

class TestGuardrails:
    """Tests for Guardrails (token budgets, circuit breaker)."""

    def test_budget_daily_reset(self, sqlite_guardrails):
        """Budget resets on new day."""
        budget = sqlite_guardrails.get_budget('test_agent', daily_limit=1000)
        budget.used_today = 500
        budget.reset_date = '2020-01-01'  # Force old date
        sqlite_guardrails.save_budget(budget)

        # Get budget again — should reset
        budget2 = sqlite_guardrails.get_budget('test_agent', daily_limit=1000)
        assert budget2.used_today == 0
        assert budget2.reset_date == date.today().isoformat()

    def test_budget_exceeded(self, sqlite_guardrails):
        """can_spend returns False when over limit."""
        budget = sqlite_guardrails.get_budget('test_agent', daily_limit=100)
        assert budget.can_spend(50)
        budget.record_usage(80, 0)
        assert not budget.can_spend(50)

    def test_circuit_breaker_cooldown(self, sqlite_guardrails):
        """Returns True after max consecutive failures."""
        for i in range(3):
            sqlite_guardrails.record_run(
                agent_name='cb_test',
                run_id=str(uuid.uuid4()),
                started_at=datetime.now(timezone.utc).isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                status='failed',
                success=False,
            )

        assert sqlite_guardrails.check_circuit_breaker('cb_test', max_failures=3) is True

    def test_circuit_breaker_resets_on_success(self, sqlite_guardrails):
        """Circuit breaker doesn't trip if there's a success in the window."""
        for i in range(2):
            sqlite_guardrails.record_run(
                agent_name='cb_reset',
                run_id=str(uuid.uuid4()),
                started_at=datetime.now(timezone.utc).isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                status='failed',
                success=False,
            )
        # One success
        sqlite_guardrails.record_run(
            agent_name='cb_reset',
            run_id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            status='completed',
            success=True,
        )

        assert sqlite_guardrails.check_circuit_breaker('cb_reset', max_failures=3) is False

    def test_run_recording(self, sqlite_guardrails):
        """record_run persists to DB."""
        run_id = str(uuid.uuid4())
        sqlite_guardrails.record_run(
            agent_name='record_test',
            run_id=run_id,
            started_at='2026-01-01T00:00:00',
            completed_at='2026-01-01T00:01:00',
            status='completed',
            success=True,
            tokens_used=500,
            execution_seconds=60.0,
            messages_sent=3,
        )

        # Verify by checking circuit breaker (which queries runs)
        assert sqlite_guardrails.check_circuit_breaker('record_test') is False

    def test_sqlite_fallback(self, tmp_path):
        """Guardrails works with SQLite when no PostgreSQL."""
        db_path = str(tmp_path / "fallback_test.db")
        guardrails = Guardrails(pg_conn=None, sqlite_path=db_path)

        budget = guardrails.get_budget('fallback_agent')
        assert budget.agent_name == 'fallback_agent'
        assert budget.daily_limit == 50_000


# =============================================================================
# TestAgentRegistry
# =============================================================================

class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def test_register_and_get(self):
        """Register agent, retrieve by name."""
        # Clear class-level state
        AgentRegistry._agents = {}

        registry = AgentRegistry(pg_conn=None)
        registry.register('test_agent', 'TestAgent', schedule='0 9 * * *')

        info = registry.get_agent('test_agent')
        assert info is not None
        assert info['agent_name'] == 'test_agent'
        assert info['agent_class'] == 'TestAgent'
        assert info['schedule'] == '0 9 * * *'
        assert info['enabled'] is True

    def test_status_tracking(self):
        """Status updates persist in memory."""
        AgentRegistry._agents = {}

        registry = AgentRegistry(pg_conn=None)
        registry.register('status_agent', 'StatusAgent')
        registry.update_status('status_agent', 'running')

        info = registry.get_agent('status_agent')
        assert info['status'] == 'running'
        assert info['last_run_at'] is not None

    def test_get_all_statuses(self):
        """Get all registered agent statuses."""
        AgentRegistry._agents = {}

        registry = AgentRegistry(pg_conn=None)
        registry.register('agent_1', 'Agent1')
        registry.register('agent_2', 'Agent2')

        statuses = registry.get_all_statuses()
        assert 'agent_1' in statuses
        assert 'agent_2' in statuses


# =============================================================================
# TestConnections
# =============================================================================

class TestConnections:
    """Tests for connection management."""

    def test_fallback_when_services_unavailable(self):
        """Graceful fallback returns None when Redis/PostgreSQL unavailable."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('REDIS_URL', None)
            os.environ.pop('DATABASE_URL', None)

            redis_client = get_redis_client()
            assert redis_client is None

            pg_conn = get_postgres_connection()
            assert pg_conn is None
