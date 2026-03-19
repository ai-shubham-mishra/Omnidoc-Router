"""
Universal File Handler for the LLM Router.
Supports ANY file type: PDF, images, video, audio, Excel, JSON, etc.
Files are stored in tmp/{runId}/ to match existing workflow conventions.
"""
import os
import re
import shutil
import logging
import mimetypes
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class FileHandler:
    """Handle file uploads for chat sessions."""

    def __init__(self, base_dir: str = "tmp"):
        self.base_dir = base_dir

    async def save_files(
        self,
        run_id: str,
        files: list,
    ) -> List[Dict[str, Any]]:
        """
        Save uploaded files to tmp/{runId}/ directory.
        Matches the existing workflow file storage convention.

        Args:
            run_id: The runId for this workflow execution
            files: List of FastAPI UploadFile objects

        Returns:
            List of file info dicts with stored_path, original_name, etc.
        """
        storage_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(storage_dir, exist_ok=True)

        stored_files = []

        for file in files:
            if not file or not file.filename:
                continue

            original_name = file.filename
            safe_name = self._sanitize_filename(original_name)
            file_path = os.path.join(storage_dir, safe_name)

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            mime_type, _ = mimetypes.guess_type(original_name)
            if not mime_type:
                mime_type = "application/octet-stream"

            stored_files.append(
                {
                    "original_name": original_name,
                    "stored_path": file_path,
                    "size_bytes": len(content),
                    "mime_type": mime_type,
                    "uploaded_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
            logger.info(f"File saved: {original_name} -> {file_path} ({len(content)} bytes)")

        return stored_files

    def cleanup_session_files(self, run_id: str):
        """Delete all files for a run."""
        session_dir = os.path.join(self.base_dir, run_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir, ignore_errors=True)
            logger.info(f"Cleaned up files for runId: {run_id}")

    def _sanitize_filename(self, filename: str) -> str:
        """Remove unsafe characters from filename, keep extension."""
        safe = re.sub(r"[^a-zA-Z0-9._\-]", "_", filename)
        return safe[:200]
