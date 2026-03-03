"""
Shared notification utilities for the agent system.

Supports Pushover (primary push delivery), Slack, and email.
All sends are wrapped in try/except — notification failure never crashes the caller.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Pushover limits
PUSHOVER_MESSAGE_LIMIT = 1024


def send_pushover(title: str, message: str, priority: int = 0) -> bool:
    """
    Send a push notification via Pushover.

    Args:
        title: Notification title (max 250 chars).
        message: Body text (max 1024 chars — truncated if longer).
        priority: -2 (silent) to 2 (emergency). 0 = normal, 1 = high.

    Returns:
        True if sent successfully, False otherwise.
    """
    app_token = os.environ.get('PUSHOVER_APP_TOKEN')
    user_key = os.environ.get('PUSHOVER_USER_KEY')

    if not app_token or not user_key:
        logger.debug("Pushover not configured (PUSHOVER_APP_TOKEN / PUSHOVER_USER_KEY not set)")
        return False

    try:
        import requests

        # Truncate message to Pushover limit
        if len(message) > PUSHOVER_MESSAGE_LIMIT:
            message = message[:PUSHOVER_MESSAGE_LIMIT - 3] + '...'

        payload = {
            'token': app_token,
            'user': user_key,
            'title': title[:250],
            'message': message,
            'priority': priority,
        }

        # Emergency priority requires retry/expire params
        if priority == 2:
            payload['retry'] = 60
            payload['expire'] = 300

        resp = requests.post(
            'https://api.pushover.net/1/messages.json',
            data=payload,
            timeout=10,
        )

        if resp.status_code == 200:
            logger.info(f"Pushover sent: {title}")
            return True
        else:
            logger.warning(f"Pushover failed ({resp.status_code}): {resp.text[:200]}")
            return False

    except Exception as e:
        logger.warning(f"Pushover send failed: {e}")
        return False


def send_alert(subject: str, message: str, severity: str = 'info'):
    """
    Send a system alert via log + Slack + email (if configured).

    Drop-in replacement for the inline send_alert() in scheduled_retraining.py.
    """
    import subprocess

    logger.log(
        logging.CRITICAL if severity == 'critical' else
        logging.ERROR if severity == 'error' else
        logging.WARNING if severity == 'warning' else logging.INFO,
        f"ALERT [{severity.upper()}]: {subject}\n{message}"
    )

    # Email alert (if configured)
    email = os.environ.get('ALERT_EMAIL')
    if email:
        try:
            subprocess.run(
                ['mail', '-s', f"[NBA Model] {subject}", email],
                input=message.encode(),
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"Failed to send email alert: {e}")

    # Slack webhook (if configured)
    webhook_url = os.environ.get('SLACK_WEBHOOK')
    if webhook_url:
        try:
            import requests
            emoji_map = {
                'critical': ':red_circle:', 'error': ':x:',
                'warning': ':warning:', 'info': ':information_source:',
            }
            emoji = emoji_map.get(severity, ':bell:')

            requests.post(webhook_url, json={
                'text': f"{emoji} *{subject}*\n{message}"
            }, timeout=10)
        except Exception as e:
            logger.warning(f"Failed to send Slack alert: {e}")


def send_briefing(formatted_text: str, briefing_date: str, play_count: int) -> bool:
    """
    Send the daily briefing via Pushover.

    Condenses the full briefing into a push-friendly summary because
    Pushover has a 1024-char message limit.

    Args:
        formatted_text: Full formatted briefing text.
        briefing_date: e.g. "2026-03-03".
        play_count: Number of actionable plays today.

    Returns:
        True if notification was sent successfully.
    """
    title = f"Daily Briefing - {briefing_date} ({play_count} play{'s' if play_count != 1 else ''})"

    # Build a condensed body that fits within 1024 chars.
    # Extract key sections from the formatted text.
    body_lines = []
    current_section = None

    for line in formatted_text.split('\n'):
        stripped = line.strip()

        # Detect section headers
        if stripped in ("YESTERDAY'S RESULTS", "ALERTS", "MARKET INTEL", "BANKROLL"):
            current_section = stripped
            continue
        if stripped.startswith("TODAY'S PLAYS"):
            current_section = 'PLAYS'
            body_lines.append(stripped)
            continue

        # Include key content
        if current_section == "YESTERDAY'S RESULTS" and stripped.startswith('Record:'):
            body_lines.insert(0, stripped)
        elif current_section == 'PLAYS' and stripped.startswith('['):
            body_lines.append(stripped)
        elif current_section == 'ALERTS' and stripped and stripped != 'ALERTS':
            body_lines.append(stripped)

    body = '\n'.join(body_lines) if body_lines else formatted_text[:800]

    try:
        return send_pushover(title, body, priority=0)
    except Exception as e:
        logger.warning(f"Failed to send briefing notification: {e}")
        return False
