"""
Universal File Handler for the LLM Router.
Supports ANY file type: PDF, images, video, audio, Excel, JSON, etc.
Files are stored in tmp/{session_id}/ as persistent context.
"""
import os
import re
import shutil
import uuid
import logging
import mimetypes
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class FileHandler:
    """Handle file uploads for chat sessions with context persistence."""

    def __init__(self, base_dir: str = "tmp"):
        self.base_dir = base_dir

    async def save_files_to_session(
        self,
        session_id: str,
        files: list,
    ) -> List[Dict[str, Any]]:
        """
        Save uploaded files to tmp/{session_id}/ directory.
        Files persist as session context (not tied to specific workflow).

        Args:
            session_id: The session ID
            files: List of FastAPI UploadFile objects

        Returns:
            List of file info dicts with file_id, stored_path, metadata, etc.
        """
        storage_dir = os.path.join(self.base_dir, session_id)
        os.makedirs(storage_dir, exist_ok=True)

        stored_files = []

        for file in files:
            if not file or not file.filename:
                continue

            original_name = file.filename
            safe_name = self._sanitize_filename(original_name)
            
            # Generate unique file_id to avoid collisions
            file_id = str(uuid.uuid4())
            file_path = os.path.join(storage_dir, f"{file_id}_{safe_name}")

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            mime_type, _ = mimetypes.guess_type(original_name)
            if not mime_type:
                mime_type = "application/octet-stream"

            stored_files.append(
                {
                    "file_id": file_id,
                    "original_name": original_name,
                    "stored_path": file_path,
                    "size_bytes": len(content),
                    "mime_type": mime_type,
                    "uploaded_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "used_in_workflows": [],  # Track which workflows used this file
                    "accessible": True,
                }
            )
            logger.info(f"📎 File saved to context: {original_name} ({len(content)} bytes)")

        return stored_files

    def get_files_for_workflow(self, session_files: List[Dict], field_type: str = "file") -> List[str]:
        """
        Get file paths from session context that match the input type.
        Returns list of stored paths accessible for workflow execution.
        
        Args:
            session_files: List of file metadata from session.uploaded_files
            field_type: Input type to filter ("file", "image", etc.)
        
        Returns:
            List of file paths ready for workflow
        """
        accessible_files = [
            f["stored_path"] 
            for f in session_files 
            if f.get("accessible", True) and os.path.exists(f.get("stored_path", ""))
        ]
        return accessible_files

    def cleanup_session_files(self, session_id: str):
        """Delete all files for a session."""
        session_dir = os.path.join(self.base_dir, session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir, ignore_errors=True)
            logger.info(f"🗑️ Cleaned up files for session: {session_id[:8]}...")

    def _sanitize_filename(self, filename: str) -> str:
        """Remove unsafe characters from filename, keep extension."""
        safe = re.sub(r"[^a-zA-Z0-9._\-]", "_", filename)
        return safe[:200]
