"""
Inter-agent event queue.

Primary: Redis Streams for ordered, persistent messaging.
Fallback: In-memory queue for tests and local dev.
"""

import json
import uuid
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

PRIORITY_ORDER = {'urgent': 0, 'high': 1, 'normal': 2, 'low': 3}


@dataclass
class Message:
    """Inter-agent message matching CLAUDE.md spec."""
    message_id: str
    timestamp: str
    sender: str
    recipient: str
    event_type: str
    priority: str = 'normal'
    payload: dict = field(default_factory=dict)
    ttl_minutes: int = 60

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def create(
        cls,
        sender: str,
        recipient: str,
        event_type: str,
        payload: dict = None,
        priority: str = 'normal',
        ttl_minutes: int = 60,
    ) -> 'Message':
        return cls(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            sender=sender,
            recipient=recipient,
            event_type=event_type,
            priority=priority,
            payload=payload or {},
            ttl_minutes=ttl_minutes,
        )


class MessageBus:
    """Redis-backed inter-agent event queue using Redis Streams."""

    def __init__(self, redis_client):
        self._redis = redis_client
        logger.info("Redis MessageBus initialized")

    def _stream_key(self, recipient: str) -> str:
        return f"agent_messages:{recipient}"

    def _payload_key(self, message_id: str) -> str:
        return f"msg:{message_id}"

    def send(self, message: Message) -> str:
        """Send a message to the recipient's stream."""
        msg_data = message.to_dict()
        payload_json = json.dumps(msg_data)

        # Store full payload with TTL
        self._redis.set(
            self._payload_key(message.message_id),
            payload_json,
            ex=message.ttl_minutes * 60,
        )

        # Add to recipient stream
        stream_data = {
            'message_id': message.message_id,
            'sender': message.sender,
            'event_type': message.event_type,
            'priority': message.priority,
            'timestamp': message.timestamp,
        }
        self._redis.xadd(self._stream_key(message.recipient), stream_data)

        # If broadcast, also add to broadcast stream
        if message.recipient == 'all':
            pass  # Already added to agent_messages:all
        else:
            # Also add to broadcast if recipient is specific
            # (broadcast messages go to agent_messages:all only)
            pass

        logger.debug(f"Message sent: {message.sender} -> {message.recipient} [{message.event_type}]")
        return message.message_id

    def receive(
        self,
        recipient: str,
        event_type: str = None,
        priority: str = None,
        count: int = 100,
    ) -> list[Message]:
        """
        Read messages for a recipient.

        Reads from both the recipient's stream and the broadcast stream.
        """
        messages = []

        # Read from recipient stream
        messages.extend(self._read_stream(self._stream_key(recipient), count))

        # Also read from broadcast stream
        if recipient != 'all':
            messages.extend(self._read_stream(self._stream_key('all'), count))

        # Filter by event_type
        if event_type:
            messages = [m for m in messages if m.event_type == event_type]

        # Filter by priority
        if priority:
            messages = [m for m in messages if m.priority == priority]

        # Sort by priority (urgent first), then by timestamp
        messages.sort(key=lambda m: (
            PRIORITY_ORDER.get(m.priority, 2),
            m.timestamp,
        ))

        return messages

    def _read_stream(self, stream_key: str, count: int) -> list[Message]:
        """Read messages from a Redis stream."""
        messages = []
        try:
            entries = self._redis.xrange(stream_key, count=count)
            for entry_id, data in entries:
                msg_id = data.get('message_id', '')
                payload_json = self._redis.get(self._payload_key(msg_id))
                if payload_json:
                    msg_data = json.loads(payload_json)
                    messages.append(Message.from_dict(msg_data))
        except Exception as e:
            logger.warning(f"Failed to read stream {stream_key}: {e}")
        return messages

    def acknowledge(self, message_id: str, consumer: str):
        """Mark a message as consumed by deleting its payload."""
        self._redis.delete(self._payload_key(message_id))
        logger.debug(f"Message {message_id} acknowledged by {consumer}")

    def get_recent(
        self,
        sender: str = None,
        minutes: int = 60,
    ) -> list[Message]:
        """Get recent messages across all streams, optionally filtered by sender."""
        # Scan all agent_messages:* streams
        messages = []
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(cursor, match='agent_messages:*', count=100)
            for key in keys:
                messages.extend(self._read_stream(key, count=200))
            if cursor == 0:
                break

        # Deduplicate by message_id
        seen = set()
        unique = []
        for m in messages:
            if m.message_id not in seen:
                seen.add(m.message_id)
                unique.append(m)

        # Filter by sender
        if sender:
            unique = [m for m in unique if m.sender == sender]

        # Filter by time
        cutoff = datetime.now(timezone.utc).timestamp() - (minutes * 60)
        result = []
        for m in unique:
            try:
                msg_ts = datetime.fromisoformat(m.timestamp).timestamp()
                if msg_ts >= cutoff:
                    result.append(m)
            except (ValueError, TypeError):
                result.append(m)

        result.sort(key=lambda m: m.timestamp)
        return result


class InMemoryMessageBus(MessageBus):
    """In-memory message bus for tests and local dev when Redis is unavailable."""

    def __init__(self):
        self._streams = defaultdict(list)
        self._payloads = {}
        self._expiry = {}
        logger.warning("Redis unavailable, using in-memory message bus")

    def send(self, message: Message) -> str:
        msg_data = message.to_dict()
        self._payloads[message.message_id] = msg_data
        self._expiry[message.message_id] = time.time() + (message.ttl_minutes * 60)
        self._streams[message.recipient].append(message.message_id)
        logger.debug(f"Message sent (in-memory): {message.sender} -> {message.recipient} [{message.event_type}]")
        return message.message_id

    def receive(
        self,
        recipient: str,
        event_type: str = None,
        priority: str = None,
        count: int = 100,
    ) -> list[Message]:
        messages = []
        now = time.time()

        # Read from recipient queue and broadcast
        streams_to_check = [recipient]
        if recipient != 'all':
            streams_to_check.append('all')

        for stream in streams_to_check:
            for msg_id in self._streams.get(stream, []):
                # Skip expired
                if msg_id in self._expiry and self._expiry[msg_id] < now:
                    continue
                # Skip acknowledged (deleted payload)
                if msg_id not in self._payloads:
                    continue
                messages.append(Message.from_dict(self._payloads[msg_id]))

        if event_type:
            messages = [m for m in messages if m.event_type == event_type]
        if priority:
            messages = [m for m in messages if m.priority == priority]

        messages.sort(key=lambda m: (
            PRIORITY_ORDER.get(m.priority, 2),
            m.timestamp,
        ))

        return messages[:count]

    def acknowledge(self, message_id: str, consumer: str):
        self._payloads.pop(message_id, None)
        logger.debug(f"Message {message_id} acknowledged by {consumer} (in-memory)")

    def get_recent(
        self,
        sender: str = None,
        minutes: int = 60,
    ) -> list[Message]:
        now = time.time()
        cutoff = now - (minutes * 60)
        messages = []
        seen = set()

        for stream_msgs in self._streams.values():
            for msg_id in stream_msgs:
                if msg_id in seen:
                    continue
                seen.add(msg_id)
                if msg_id not in self._payloads:
                    continue
                if msg_id in self._expiry and self._expiry[msg_id] < now:
                    continue
                msg = Message.from_dict(self._payloads[msg_id])
                try:
                    msg_ts = datetime.fromisoformat(msg.timestamp).timestamp()
                    if msg_ts < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
                messages.append(msg)

        if sender:
            messages = [m for m in messages if m.sender == sender]

        messages.sort(key=lambda m: m.timestamp)
        return messages
