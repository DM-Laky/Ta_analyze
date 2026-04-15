from __future__ import annotations

import asyncio

from core.orchestrator import Orchestrator


async def _main() -> None:
    orchestrator = Orchestrator()
    await orchestrator.start()


if __name__ == "__main__":
    asyncio.run(_main())
