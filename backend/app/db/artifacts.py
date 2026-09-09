import io
from typing import Any

from bson import ObjectId
from gridfs import AsyncGridFSBucket


class ArtifactStore:
    def __init__(self, database: Any, bucket_name: str) -> None:
        self.bucket = AsyncGridFSBucket(database, bucket_name=bucket_name)

    async def put(self, filename: str, content: bytes, metadata: dict[str, Any]) -> str:
        artifact_id = await self.bucket.upload_from_stream(
            filename,
            content,
            metadata=metadata,
        )
        return str(artifact_id)

    async def read(self, artifact_id: str) -> bytes:
        target = io.BytesIO()
        await self.bucket.download_to_stream(ObjectId(artifact_id), target)
        return target.getvalue()
