import asyncio
from typing import Any

from pymongo import AsyncMongoClient

from app.core.config import get_settings

_clients: dict[int, AsyncMongoClient[dict[str, Any]]] = {}


def get_mongo_client() -> AsyncMongoClient[dict[str, Any]]:
    loop_id = id(asyncio.get_running_loop())
    client = _clients.get(loop_id)
    if client is None:
        client = AsyncMongoClient(get_settings().mongodb_uri, serverSelectionTimeoutMS=3000)
        _clients[loop_id] = client
    return client


def get_database() -> Any:
    return get_mongo_client()[get_settings().mongodb_database]


async def close_mongo_client() -> None:
    loop_id = id(asyncio.get_running_loop())
    client = _clients.pop(loop_id, None)
    if client is not None:
        await client.close()


async def check_connection() -> bool:
    try:
        await get_mongo_client().admin.command("ping")
        return True
    except Exception:
        return False
