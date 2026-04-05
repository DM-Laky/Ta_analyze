"""
core/session_manager.py
========================
Detects active trading sessions (London, New York, Overlap).
Determines whether the bot should run deep analysis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from config import config
from utils.logger import log


@dataclass
class SessionInfo:
    london: bool
    new_york: bool
    overlap: bool          # London + NY overlap = highest volume
    name: str
    emoji: str
    is_active: bool        # True if any major session is open

    def __str__(self) -> str:
        return f"{self.emoji} {self.name}"


class SessionManager:
    """Checks which market sessions are currently active."""

    def get_current_session(self, dt: Optional[datetime] = None) -> SessionInfo:
        """
        Returns session info for the given datetime (UTC).
        If `dt` is None, uses current UTC time.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        hour = dt.hour

        london = config.LONDON_START_UTC <= hour < config.LONDON_END_UTC
        new_york = config.NY_START_UTC <= hour < config.NY_END_UTC
        overlap = london and new_york

        if overlap:
            name = "London + New York OVERLAP"
            emoji = "🔥"
        elif london:
            name = "London Session"
            emoji = "🇬🇧"
        elif new_york:
            name = "New York Session"
            emoji = "🗽"
        else:
            name = "Off-Session (Low Volume)"
            emoji = "😴"

        is_active = london or new_york

        return SessionInfo(
            london=london,
            new_york=new_york,
            overlap=overlap,
            name=name,
            emoji=emoji,
            is_active=is_active,
        )

    def should_run_analysis(self, dt: Optional[datetime] = None) -> bool:
        """Return True if we are in a major session."""
        session = self.get_current_session(dt)
        return session.is_active

    def session_quality(self, dt: Optional[datetime] = None) -> float:
        """
        Return a 0–1 score for session quality.
        Overlap = 1.0, single session = 0.7, off = 0.0
        """
        session = self.get_current_session(dt)
        if session.overlap:
            return 1.0
        if session.is_active:
            return 0.7
        return 0.0

    def log_session(self, dt: Optional[datetime] = None):
        session = self.get_current_session(dt)
        quality = self.session_quality(dt)
        log.info(
            "📅 Session: %s | Quality: %.0f%% | Active: %s",
            session, quality * 100, session.is_active
        )


session_manager = SessionManager()
