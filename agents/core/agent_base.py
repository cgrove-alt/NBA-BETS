"""
Abstract base class every agent inherits.

Provides lifecycle management, LLM calls, token tracking,
shadow mode, and circuit breaker logic.
"""

import json
import uuid
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from .message_bus import MessageBus, InMemoryMessageBus, Message
from .guardrails import Guardrails

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    IDLE = 'idle'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    COOLDOWN = 'cooldown'


@dataclass
class AgentResult:
    """Result of an agent execution."""
    agent_name: str
    status: str
    started_at: str
    completed_at: str
    run_id: str
    messages_sent: int = 0
    tokens_used: int = 0
    errors: list = field(default_factory=list)
    payload: dict = field(default_factory=dict)
    reasoning: str = ''


class AgentBase(ABC):
    """
    Abstract base class for all agents.

    Subclasses must implement:
        - AGENT_NAME: str
        - DAILY_TOKEN_BUDGET: int
        - MAX_EXECUTION_SECONDS: int
        - run() -> dict: Core agent logic
    """

    AGENT_NAME: str = 'base'
    DAILY_TOKEN_BUDGET: int = 50_000
    MAX_EXECUTION_SECONDS: int = 300
    MAX_CONSECUTIVE_FAILURES: int = 3

    def __init__(
        self,
        message_bus: MessageBus = None,
        guardrails: Guardrails = None,
        shadow_mode: bool = False,
    ):
        self.message_bus = message_bus or InMemoryMessageBus()
        self.guardrails = guardrails or Guardrails()
        self.shadow_mode = shadow_mode
        self._tokens_used = 0
        self._messages_sent = 0
        self._run_id = ''
        self._llm_client = None

    def execute(self) -> AgentResult:
        """
        Full agent lifecycle: run -> report -> cleanup.

        Returns AgentResult with status, timing, token usage, etc.
        """
        self._run_id = str(uuid.uuid4())
        self._tokens_used = 0
        self._messages_sent = 0
        started_at = datetime.now(timezone.utc).isoformat()
        errors = []

        logger.info(f"[{self.AGENT_NAME}] Starting execution (run_id={self._run_id}, shadow={self.shadow_mode})")

        # Check circuit breaker
        if self.guardrails.check_circuit_breaker(self.AGENT_NAME, self.MAX_CONSECUTIVE_FAILURES):
            logger.warning(f"[{self.AGENT_NAME}] Circuit breaker OPEN — agent in COOLDOWN")
            completed_at = datetime.now(timezone.utc).isoformat()
            result = AgentResult(
                agent_name=self.AGENT_NAME,
                status=AgentStatus.COOLDOWN.value,
                started_at=started_at,
                completed_at=completed_at,
                run_id=self._run_id,
                errors=['Circuit breaker open — too many consecutive failures'],
            )
            self.guardrails.record_run(
                agent_name=self.AGENT_NAME, run_id=self._run_id,
                started_at=started_at, completed_at=completed_at,
                status='cooldown', success=False,
            )
            return result

        try:
            # Run core logic
            run_output = self.run()

            # Report results (send messages)
            self.report(run_output)

            # Cleanup
            self.cleanup()

            completed_at = datetime.now(timezone.utc).isoformat()
            execution_seconds = (
                datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)
            ).total_seconds()

            result = AgentResult(
                agent_name=self.AGENT_NAME,
                status=AgentStatus.COMPLETED.value,
                started_at=started_at,
                completed_at=completed_at,
                run_id=self._run_id,
                messages_sent=self._messages_sent,
                tokens_used=self._tokens_used,
                payload=run_output if isinstance(run_output, dict) else {'output': run_output},
                reasoning=run_output.get('reasoning', '') if isinstance(run_output, dict) else '',
            )

            self.guardrails.record_run(
                agent_name=self.AGENT_NAME, run_id=self._run_id,
                started_at=started_at, completed_at=completed_at,
                status='completed', success=True,
                tokens_used=self._tokens_used,
                execution_seconds=execution_seconds,
                messages_sent=self._messages_sent,
                payload=run_output if isinstance(run_output, dict) else {'output': run_output},
            )

            logger.info(
                f"[{self.AGENT_NAME}] Completed (tokens={self._tokens_used}, "
                f"messages={self._messages_sent}, seconds={execution_seconds:.1f})"
            )
            return result

        except Exception as e:
            completed_at = datetime.now(timezone.utc).isoformat()
            execution_seconds = (
                datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)
            ).total_seconds()

            errors.append(str(e))
            logger.error(f"[{self.AGENT_NAME}] Failed: {e}", exc_info=True)

            result = AgentResult(
                agent_name=self.AGENT_NAME,
                status=AgentStatus.FAILED.value,
                started_at=started_at,
                completed_at=completed_at,
                run_id=self._run_id,
                messages_sent=self._messages_sent,
                tokens_used=self._tokens_used,
                errors=errors,
            )

            self.guardrails.record_run(
                agent_name=self.AGENT_NAME, run_id=self._run_id,
                started_at=started_at, completed_at=completed_at,
                status='failed', success=False,
                tokens_used=self._tokens_used,
                execution_seconds=execution_seconds,
                errors=errors,
            )

            return result

    @abstractmethod
    def run(self) -> dict:
        """Core agent logic. Subclasses must implement."""
        raise NotImplementedError

    def report(self, run_output: dict):  # noqa: B027
        """Send messages to the bus based on run output. Override in subclasses."""

    def cleanup(self):  # noqa: B027
        """Optional cleanup after run. Override in subclasses."""

    def call_llm(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
        response_json: bool = True,
    ) -> str:
        """
        Call LLM for reasoning. Default: Gemini 2.0 Flash.

        Falls back gracefully if GEMINI_API_KEY is not set.

        Args:
            system_prompt: System prompt for the LLM
            user_message: User message with data to analyze
            max_tokens: Maximum tokens in response
            response_json: Whether to request JSON output

        Returns:
            LLM response text, or empty string on failure
        """
        import os

        # Check budget
        budget = self.guardrails.get_budget(self.AGENT_NAME, self.DAILY_TOKEN_BUDGET)
        estimated_input = len(system_prompt + user_message) // 4  # rough estimate
        if not budget.can_spend(estimated_input + max_tokens):
            logger.warning(f"[{self.AGENT_NAME}] Token budget exceeded — skipping LLM call")
            return ''

        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            logger.warning(f"[{self.AGENT_NAME}] GEMINI_API_KEY not set — skipping LLM call")
            return ''

        try:
            if self._llm_client is None:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                generation_config = {}
                if response_json:
                    generation_config['response_mime_type'] = 'application/json'
                self._llm_client = genai.GenerativeModel(
                    'gemini-2.0-flash',
                    generation_config=generation_config,
                    system_instruction=system_prompt,
                )

            response = self._llm_client.generate_content(
                user_message,
                generation_config={'max_output_tokens': max_tokens},
            )

            # Track tokens
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage_metadata'):
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

            self._tokens_used += (input_tokens + output_tokens)
            budget.record_usage(input_tokens, output_tokens)
            self.guardrails.save_budget(budget)

            result_text = response.text if hasattr(response, 'text') else ''
            logger.debug(f"[{self.AGENT_NAME}] LLM call: {input_tokens} in, {output_tokens} out")
            return result_text

        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] LLM call failed: {e}")
            return ''

    def send_message(
        self,
        recipient: str,
        event_type: str,
        payload: dict,
        priority: str = 'normal',
        ttl_minutes: int = 60,
    ):
        """Send a message to the bus. No-op in shadow mode."""
        if self.shadow_mode:
            logger.debug(f"[{self.AGENT_NAME}] Shadow mode — suppressing message to {recipient}")
            return

        msg = Message.create(
            sender=self.AGENT_NAME,
            recipient=recipient,
            event_type=event_type,
            payload=payload,
            priority=priority,
            ttl_minutes=ttl_minutes,
        )
        self.message_bus.send(msg)
        self._messages_sent += 1

    def get_messages(self, event_type: str = None) -> list[Message]:
        """Read messages addressed to this agent."""
        return self.message_bus.receive(self.AGENT_NAME, event_type=event_type)
