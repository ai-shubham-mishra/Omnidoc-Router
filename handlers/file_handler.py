"""
Universal File Handler for the LLM Router.
Supports ANY file type: PDF, images, video, audio, Excel, JSON, etc.
Files are stored in Azure Blob Storage or local (environment-driven).
Hierarchical path: knowledge-base/<orgId>/<userId>/<sessionId>/<runId>/<stage>/...
"""
import os
import re
import tempfile
import logging
import mimetypes
from datetime import datetime
from typing import List, Dict, Any

from utils.storage_factory import get_storage_backend
from utils.config import STORAGE_BACKEND

logger = logging.getLogger(__name__)


class FileHandler:
    """Handle file uploads for chat sessions with cloud/local storage support."""

    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

    def __init__(self):
        self.storage = get_storage_backend()
        self.storage_type = STORAGE_BACKEND
        logger.info(f"📦 FileHandler initialized with {self.storage_type} storage")

    async def save_files_to_session(
        self,
        session_id: str,
        files: list,
        user_id: str = None,
        org_id: str = None,
        run_id: str = None,
        stage: str = "input",
    ) -> List[Dict[str, Any]]:
        """
        Save uploaded files to storage using hierarchical paths.

        Args:
            session_id: The session / chat ID
            files: List of FastAPI UploadFile objects
            user_id: User ID (from JWT)
            org_id: Organization ID (from JWT)
            run_id: Workflow run ID (optional, defaults to session_id)
            stage: File lifecycle stage ('input', 'intermediate', 'output')

        Returns:
            List of file info dicts
        """
        stored_files = []

        for file in files:
            if not file or not file.filename:
                continue

            original_name = file.filename
            content = await file.read()
            file_size = len(content)

            if file_size > self.MAX_FILE_SIZE_BYTES:
                size_mb = file_size / (1024 * 1024)
                limit_mb = self.MAX_FILE_SIZE_BYTES / (1024 * 1024)
                logger.warning(f"⚠️ File '{original_name}' ({size_mb:.1f}MB) exceeds {limit_mb}MB limit - skipped")
                continue

            mime_type, _ = mimetypes.guess_type(original_name)
            if not mime_type:
                mime_type = "application/octet-stream"

            metadata = {"mime_type": mime_type}

            try:
                upload_result = self.storage.upload_file(
                    file_data=content,
                    original_filename=original_name,
                    org_id=org_id,
                    user_id=user_id,
                    session_id=session_id,
                    run_id=run_id or session_id,
                    stage=stage,
                    metadata=metadata,
                )

                file_info = {
                    "file_id": upload_result["file_id"],
                    "original_name": original_name,
                    "stored_path": upload_result["blob_path"],
                    "blob_url": upload_result.get("blob_url"),
                    "extension": upload_result["extension"],
                    "size_bytes": upload_result["size_bytes"],
                    "mime_type": mime_type,
                    "uploaded_at": upload_result["uploaded_at"],
                    "stage": upload_result.get("stage", stage),
                    "used_in_workflows": [],
                    "accessible": True,
                }

                stored_files.append(file_info)
                logger.info(f"📎 File uploaded: {original_name} ({file_size / 1024:.1f}KB) [{stage}]")

            except Exception as e:
                logger.error(f"❌ Failed to upload {original_name}: {e}")
                continue

        return stored_files

    def get_files_for_workflow(
        self,
        session_files: List[Dict],
        field_type: str = "file",
    ) -> List[str]:
        """
        Get local file paths from session context for workflow execution.
        Downloads from blob storage to temp if needed.
        """
        local_paths = []

        for f in session_files:
            if not f.get("accessible", True):
                continue

            stored_path = f.get("stored_path", "")

            if self.storage_type == "azure" and not os.path.exists(stored_path):
                try:
                    local_path = self.storage.download_to_local(stored_path)
                    local_paths.append(local_path)
                    logger.info(f"📥 Downloaded blob to temp: {stored_path} → {local_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to download blob {stored_path}: {e}")
                    continue
            elif os.path.exists(stored_path):
                local_paths.append(stored_path)
            else:
                logger.warning(f"⚠️ File not accessible: {stored_path}")

        return local_paths

    def _sanitize_filename(self, filename: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._\-]", "_", filename)
        return safe[:200]

