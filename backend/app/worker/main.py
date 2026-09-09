import asyncio
import signal

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.client import check_connection, close_mongo_client
from app.worker.orchestrator import WorkerOrchestrator


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if not await check_connection():
        raise RuntimeError("MONGODB_ERROR: worker requires a persistent MongoDB connection")
    orchestrator = WorkerOrchestrator(settings)
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, orchestrator.stop_event.set)
        except NotImplementedError:
            signal.signal(signal_name, lambda *_: orchestrator.stop_event.set())
    try:
        await orchestrator.run()
    finally:
        await orchestrator.shutdown()
        await close_mongo_client()


if __name__ == "__main__":
    asyncio.run(main())
