"""
Agent Core - Framework components for autonomous agents.

Provides:
    - connections: Redis + PostgreSQL connection management
    - message_bus: Inter-agent event queue (Redis Streams + in-memory fallback)
    - guardrails: Token budgets, circuit breaker, execution limits
    - agent_base: Abstract base class with lifecycle management
    - agent_registry: Central registry tracking all agents
    - agent_runner: CLI entry point for running agents
"""

from .agent_base import AgentBase, AgentResult, AgentStatus
from .message_bus import MessageBus, InMemoryMessageBus, Message
from .guardrails import Guardrails, TokenBudget
from .agent_registry import AgentRegistry
from .connections import get_redis_client, get_postgres_connection, ensure_agent_schema

__all__ = [
    'AgentBase',
    'AgentResult',
    'AgentStatus',
    'MessageBus',
    'InMemoryMessageBus',
    'Message',
    'Guardrails',
    'TokenBudget',
    'AgentRegistry',
    'get_redis_client',
    'get_postgres_connection',
    'ensure_agent_schema',
]
