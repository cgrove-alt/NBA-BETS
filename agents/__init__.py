"""
Agents - Autonomous AI Agent System for NBA Betting Model

This package provides the agent framework and individual agents that add
AI-powered reasoning on top of the deterministic prediction pipeline.

Agents reason. Pipelines execute.

Core Framework:
    - AgentBase: Abstract base class with lifecycle management
    - MessageBus: Redis-backed inter-agent communication
    - Guardrails: Token budgets, circuit breaker, execution limits
    - AgentRegistry: Central registry of all agents

Agents:
    - PreGameIntelAgent: Pre-game intelligence (injuries, lineups, context)
    - PostGameAnalysisAgent: Post-game analysis (miss analysis, patterns)

Usage:
    from agents.core import AgentBase, MessageBus, Guardrails
    from agents.pregame import PreGameIntelAgent
    from agents.postgame import PostGameAnalysisAgent
"""

from agents.core.agent_base import AgentBase, AgentResult, AgentStatus
from agents.core.message_bus import MessageBus, InMemoryMessageBus, Message
from agents.core.guardrails import Guardrails, TokenBudget
from agents.core.agent_registry import AgentRegistry

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
]
