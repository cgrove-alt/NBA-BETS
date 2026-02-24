"""
CLI entry point for running agents.

Usage:
    python -m agents.core.agent_runner --agent pregame [--shadow] [--date YYYY-MM-DD]
    python -m agents.core.agent_runner --agent postgame [--date YYYY-MM-DD]
    python -m agents.core.agent_runner --status
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents.core.connections import get_redis_client, get_postgres_connection, ensure_agent_schema
from agents.core.message_bus import MessageBus, InMemoryMessageBus
from agents.core.guardrails import Guardrails
from agents.core.agent_registry import AgentRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agent name -> (module_path, class_name, schedule)
AGENT_CATALOG = {
    'pregame': ('agents.pregame.pregame_agent', 'PreGameIntelAgent', '0 11,17 * * *'),
    'postgame': ('agents.postgame.postgame_agent', 'PostGameAnalysisAgent', '0 1 * * *'),
}


def _load_agent_class(module_path: str, class_name: str):
    """Dynamically import an agent class."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _setup_infrastructure():
    """Set up message bus and guardrails with appropriate backends."""
    # Redis
    redis_client = get_redis_client()
    if redis_client:
        message_bus = MessageBus(redis_client)
    else:
        message_bus = InMemoryMessageBus()

    # PostgreSQL
    pg_conn = get_postgres_connection()
    if pg_conn:
        ensure_agent_schema(pg_conn)

    guardrails = Guardrails(pg_conn=pg_conn)

    return message_bus, guardrails, pg_conn


def run_agent(agent_name: str, shadow: bool = False, target_date: str = None):
    """Run a single agent."""
    if agent_name not in AGENT_CATALOG:
        logger.error(f"Unknown agent: {agent_name}. Available: {list(AGENT_CATALOG.keys())}")
        return 1

    module_path, class_name, schedule = AGENT_CATALOG[agent_name]
    message_bus, guardrails, pg_conn = _setup_infrastructure()

    # Register agent
    registry = AgentRegistry(pg_conn=pg_conn)
    registry.register(agent_name, class_name, schedule=schedule)

    # Load and instantiate
    AgentClass = _load_agent_class(module_path, class_name)

    kwargs = {
        'message_bus': message_bus,
        'guardrails': guardrails,
        'shadow_mode': shadow,
    }
    if target_date:
        kwargs['target_date'] = target_date

    agent = AgentClass(**kwargs)

    print("=" * 60)
    print(f"AGENT: {agent_name.upper()}")
    print(f"Shadow mode: {shadow}")
    if target_date:
        print(f"Target date: {target_date}")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)

    # Execute
    registry.update_status(agent_name, 'running')
    result = agent.execute()
    registry.update_status(agent_name, result.status, {
        'run_id': result.run_id,
        'tokens_used': result.tokens_used,
        'messages_sent': result.messages_sent,
        'errors': result.errors,
    })

    # Print result
    print()
    print(f"Status: {result.status}")
    print(f"Run ID: {result.run_id}")
    print(f"Tokens used: {result.tokens_used}")
    print(f"Messages sent: {result.messages_sent}")
    if result.errors:
        print(f"Errors: {result.errors}")
    if result.reasoning:
        print(f"Reasoning: {result.reasoning[:500]}")
    print()
    print(f"Completed at: {result.completed_at}")
    print("=" * 60)

    return 0 if result.status == 'completed' else 1


def show_status():
    """Show status of all registered agents."""
    _, _, pg_conn = _setup_infrastructure()
    registry = AgentRegistry(pg_conn=pg_conn)

    # Register all known agents
    for name, (mod, cls, schedule) in AGENT_CATALOG.items():
        registry.register(name, cls, schedule=schedule)

    statuses = registry.get_all_statuses()

    print("=" * 60)
    print("AGENT STATUS")
    print("=" * 60)

    for name, info in statuses.items():
        enabled = "ENABLED" if info['enabled'] else "DISABLED"
        print(f"  {name:12s}  {info['status']:12s}  {enabled}  schedule: {info.get('schedule', 'N/A')}")
        if info.get('last_run_at'):
            print(f"               last run: {info['last_run_at']}")

    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(description='Run NBA Betting Model agents')
    parser.add_argument(
        '--agent', '-a',
        type=str,
        choices=list(AGENT_CATALOG.keys()),
        help='Agent to run',
    )
    parser.add_argument(
        '--shadow',
        action='store_true',
        help='Run in shadow mode (no messages sent to bus)',
    )
    parser.add_argument(
        '--date', '-d',
        type=str,
        default=None,
        help='Target date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show status of all agents',
    )

    args = parser.parse_args()

    if args.status:
        sys.exit(show_status())
    elif args.agent:
        sys.exit(run_agent(args.agent, shadow=args.shadow, target_date=args.date))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
