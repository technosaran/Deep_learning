"""
alerts.py
---------
Alert system for shelf AI events.

Supports:
  - Console / logging output (always enabled)
  - Telegram bot messages (optional)
  - Email via SMTP (optional)

Alerts are de-duplicated using a cooldown window so that a repeated
low-stock condition does not flood operators.
"""

from __future__ import annotations

import logging
import os
import smtplib
import time
from email.mime.text import MIMEText
from typing import Dict

import yaml

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Sends alerts when stock issues or misplacements are detected.

    Parameters
    ----------
    thresholds_path : str
        Path to ``thresholds.yaml`` (contains alert configuration).
    """

    def __init__(self, thresholds_path: str) -> None:
        with open(thresholds_path) as f:
            cfg = yaml.safe_load(f)

        alert_cfg = cfg.get("alerts", {})
        self._cooldown: int = alert_cfg.get("cooldown_seconds", 300)

        # Telegram
        tg = alert_cfg.get("telegram", {})
        self._tg_enabled: bool = tg.get("enabled", False)
        self._tg_token: str = os.getenv("TELEGRAM_BOT_TOKEN", tg.get("bot_token", ""))
        self._tg_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", tg.get("chat_id", ""))

        # Email
        em = alert_cfg.get("email", {})
        self._email_enabled: bool = em.get("enabled", False)
        self._smtp_host: str = em.get("smtp_host", "smtp.gmail.com")
        self._smtp_port: int = em.get("smtp_port", 587)
        self._email_sender: str = os.getenv("EMAIL_SENDER", em.get("sender", ""))
        self._email_password: str = os.getenv("EMAIL_PASSWORD", em.get("password", ""))
        self._email_recipient: str = os.getenv(
            "EMAIL_RECIPIENT", em.get("recipient", "")
        )

        # Cooldown tracker: alert_key -> last_sent_timestamp
        self._last_sent: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, subject: str, message: str, alert_key: str = "") -> None:
        """
        Emit an alert (subject + message body).

        Respects the cooldown window – if *alert_key* was sent recently
        the call is silently dropped.

        Parameters
        ----------
        subject :
            Short one-line description (used as email subject / Telegram title).
        message :
            Full message body.
        alert_key :
            Unique key for de-duplication (e.g. ``"shelf_b:colgate:out_of_stock"``).
            If empty, cooldown is not applied.
        """
        if alert_key:
            now = time.time()
            last = self._last_sent.get(alert_key, 0.0)
            if now - last < self._cooldown:
                logger.debug("Alert '%s' suppressed (cooldown).", alert_key)
                return
            self._last_sent[alert_key] = now

        # Always log to console
        logger.warning("[ALERT] %s | %s", subject, message)

        if self._tg_enabled:
            self._send_telegram(f"*{subject}*\n{message}")

        if self._email_enabled:
            self._send_email(subject, message)

    def send_report(self, report_text: str) -> None:
        """Send a full shelf report (e.g. daily summary)."""
        self.send(
            subject="Shelf AI – Daily Inventory Report",
            message=report_text,
            alert_key="daily_report",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_telegram(self, text: str) -> None:
        """Send a Telegram message via the Bot API."""
        try:
            import urllib.request
            import urllib.parse
            import json

            url = f"https://api.telegram.org/bot{self._tg_token}/sendMessage"
            payload = json.dumps(
                {"chat_id": self._tg_chat_id, "text": text, "parse_mode": "Markdown"}
            ).encode()
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.error("Telegram alert failed: HTTP %s", resp.status)
        except Exception as exc:  # noqa: BLE001
            logger.error("Telegram alert error: %s", exc)

    def _send_email(self, subject: str, body: str) -> None:
        """Send an email alert via SMTP."""
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self._email_sender
            msg["To"] = self._email_recipient
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.starttls()
                server.login(self._email_sender, self._email_password)
                server.send_message(msg)
        except Exception as exc:  # noqa: BLE001
            logger.error("Email alert error: %s", exc)
