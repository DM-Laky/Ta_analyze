"""
telegram/bot.py
================
Async Telegram bot using python-telegram-bot v20+.
Sends text messages and photo (chart) messages.
Queue-based to avoid flooding Telegram API.
"""

from __future__ import annotations

import asyncio
import io
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import config
from utils.logger import log

try:
    from telegram import Bot, InputFile
    from telegram.error import TelegramError, RetryAfter
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    log.warning("python-telegram-bot not installed — bot alerts disabled")


class MsgType(Enum):
    TEXT  = "text"
    PHOTO = "photo"


@dataclass
class TelegramMessage:
    msg_type: MsgType
    text: str
    photo: Optional[io.BytesIO] = None
    chat_id: Optional[str] = None


class TelegramBot:
    """
    Runs a background thread with an async event loop.
    Caller just calls .send_text() or .send_photo() — non-blocking.
    Messages are queued and sent with retry/backoff.
    """

    def __init__(self):
        self._token = config.TELEGRAM_BOT_TOKEN
        self._chat_id = config.TELEGRAM_CHAT_ID
        self._queue: queue.Queue[TelegramMessage] = queue.Queue(maxsize=100)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bot: Optional[any] = None
        self._sent_count = 0
        self._error_count = 0

    # ── Public API (thread-safe, non-blocking) ────────────────────────────────

    def send_text(
        self,
        text: str,
        chat_id: Optional[str] = None,
        disable_preview: bool = True,
    ):
        """Queue a text message."""
        msg = TelegramMessage(
            msg_type=MsgType.TEXT,
            text=text,
            chat_id=chat_id or self._chat_id,
        )
        self._enqueue(msg)

    def send_photo(
        self,
        photo: io.BytesIO,
        caption: str = "",
        chat_id: Optional[str] = None,
    ):
        """Queue a photo (chart) with caption."""
        msg = TelegramMessage(
            msg_type=MsgType.PHOTO,
            text=caption,
            photo=photo,
            chat_id=chat_id or self._chat_id,
        )
        self._enqueue(msg)

    def _enqueue(self, msg: TelegramMessage):
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            log.warning("Telegram queue full — message dropped")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if not TELEGRAM_AVAILABLE:
            log.warning("Telegram bot disabled (package not available)")
            return
        if not self._token or not self._chat_id:
            log.warning("Telegram token/chat_id not configured — alerts disabled")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="TelegramBot", daemon=True
        )
        self._thread.start()
        log.info("✅ Telegram bot started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Telegram bot stopped. Sent=%d Errors=%d",
                  self._sent_count, self._error_count)

    # ── Background loop ───────────────────────────────────────────────────────

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_worker())

    async def _async_worker(self):
        self._bot = Bot(token=self._token)
        log.info("Telegram async worker started | Chat: %s", self._chat_id)

        while self._running:
            try:
                # Non-blocking get with 1s timeout
                try:
                    msg = self._queue.get(timeout=1)
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue

                await self._dispatch(msg)
                self._queue.task_done()
                await asyncio.sleep(0.5)  # 0.5s between messages (anti-flood)

            except Exception as exc:
                log.error("Telegram worker error: %s", exc)
                await asyncio.sleep(5)

    async def _dispatch(self, msg: TelegramMessage, retry: int = 3):
        for attempt in range(retry):
            try:
                chat = msg.chat_id or self._chat_id

                if msg.msg_type == MsgType.TEXT:
                    await self._bot.send_message(
                        chat_id=chat,
                        text=msg.text,
                        parse_mode=ParseMode.MARKDOWN_V2,
                        disable_web_page_preview=True,
                    )

                elif msg.msg_type == MsgType.PHOTO and msg.photo:
                    msg.photo.seek(0)
                    await self._bot.send_photo(
                        chat_id=chat,
                        photo=msg.photo,
                        caption=msg.text[:1024] if msg.text else None,
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )

                self._sent_count += 1
                log.debug("Telegram message sent (total=%d)", self._sent_count)
                return

            except RetryAfter as e:
                wait = e.retry_after + 1
                log.warning("Telegram flood control — waiting %ds", wait)
                await asyncio.sleep(wait)

            except TelegramError as e:
                log.error("TelegramError (attempt %d/%d): %s", attempt + 1, retry, e)
                if attempt < retry - 1:
                    await asyncio.sleep(2 ** attempt)

            except Exception as e:
                log.error("Unexpected error sending Telegram message: %s", e)
                self._error_count += 1
                return

        self._error_count += 1
        log.error("Failed to send Telegram message after %d attempts", retry)

    # ── Convenience methods ───────────────────────────────────────────────────

    def send_alert_with_chart(
        self,
        text: str,
        chart: Optional[io.BytesIO],
        caption: Optional[str] = None,
    ):
        """Send chart first (with short caption), then full text."""
        if chart:
            self.send_photo(chart, caption=caption or "")
        self.send_text(text)

    @property
    def is_configured(self) -> bool:
        return bool(self._token and self._chat_id)


telegram_bot = TelegramBot()
