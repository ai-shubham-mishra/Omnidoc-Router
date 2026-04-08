"""
Universal File Handler for the LLM Router.
Supports ANY file type: PDF, images, video, audio, Excel, JSON, etc.
Files are stored in Azure Blob Storage or local (environment-driven).
Hierarchical path: knowledge-base/<orgId>/<userId>/<sessionId>/<runId>/<stage>/...

Pinecone Integration:
- Extracts text content from uploaded files
- Embeds content using Pinecone's llama-text-embed-v2 model
- Stores vectors with metadata for semantic search
"""
import os
import re
import tempfile
import logging
import mimetypes
from datetime import datetime
from typing import List, Dict, Any, Optional

from utils.storage_factory import get_storage_backend
from utils.config import STORAGE_BACKEND
from components.PineconeClient import pinecone_client
from components.TextExtractor import text_extractor

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
                    run_id=run_id,
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
                logger.info(f"File uploaded: {original_name} ({file_size / 1024:.1f}KB) [{stage}]")
                
                # Embed file content in Pinecone using in-memory bytes
                self._embed_file_in_pinecone(
                    file_info=file_info,
                    file_bytes=content,
                    session_id=session_id,
                    user_id=user_id,
                    org_id=org_id,
                    run_id=run_id,
                )

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
    
    def _embed_file_in_pinecone(
        self,
        file_info: Dict[str, Any],
        file_bytes: bytes,
        session_id: str,
        user_id: str,
        org_id: str,
        run_id: Optional[str] = None,
    ):
        """Extract text from in-memory bytes and embed in Pinecone."""
        try:
            mime_type = file_info.get("mime_type", "")
            text_content = text_extractor.extract_text_from_bytes(file_bytes, mime_type)
            
            if not text_content:
                logger.debug(f"No text content extracted from {file_info['original_name']}")
                return
            
            success = pinecone_client.embed_and_upsert_file(
                session_id=session_id,
                file_id=file_info["file_id"],
                filename=file_info["original_name"],
                content=text_content,
                user_id=user_id,
                org_id=org_id,
                run_id=run_id,
                mime_type=mime_type,
                classification=file_info.get("classification"),
            )
            
            if success:
                logger.info(f"🔍 File embedded in Pinecone: {file_info['original_name']}")
        
        except Exception as e:
            logger.error(f"Failed to embed file in Pinecone: {e}")

