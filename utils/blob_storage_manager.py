"""
Azure Blob Storage Manager for OmniDoc
Handles all file upload/download/list operations using Azure Blob Storage.

Hierarchical path structure:
  knowledge-base/<orgId>/<userId>/<sessionId>/<runId>/<stage>/<fileId>_<sanitized_name>.<ext>

Stages: input | intermediate | output
"""
import os
import re
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    generate_blob_sas,
    BlobSasPermissions,
)
from azure.core.exceptions import ResourceNotFoundError, AzureError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Valid lifecycle stages
VALID_STAGES = ("input", "intermediate", "output")


def _sanitize(name: str, max_len: int = 120) -> str:
    """Remove unsafe characters from a name segment."""
    safe = re.sub(r"[^a-zA-Z0-9._\-]", "_", name)
    return safe[:max_len]


class BlobStorageManager:
    """Manages file storage in Azure Blob Storage with hierarchical org/user/session/run paths."""

    def __init__(self):
        """Initialize Azure Blob Storage client using connection string from env."""
        self.connection_string = os.getenv("AZ_BLOB_CONN_STRING")
        self.container_name = os.getenv("AZ_BLOB_CONTAINER_NAME", "knowledge-base")
        self.account_name = os.getenv("AZ_BLOB_STORAGE_ACCOUNT_NAME", "omnidocdev")
        self.account_key = os.getenv("AZ_BLOB_CONN_KEY")
        self.base_directory = os.getenv("AZ_BLOB_BASE_DIRECTORY", "knowledge-base")

        if not self.connection_string:
            raise ValueError("AZ_BLOB_CONN_STRING not found in environment variables")

        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
            self.container_client.exists()
            logger.info(f"✅ Connected to Azure Blob Storage: {self.container_name}")
            logger.info(f"   Account: {self.account_name}, Base dir: {self.base_directory}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Azure Blob Storage: {e}")
            raise

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _build_blob_path(
        self,
        org_id: str,
        user_id: str,
        session_id: str,
        run_id: str,
        stage: str,
        file_id: str,
        original_filename: str,
    ) -> str:
        """
        Build hierarchical blob path:
        knowledge-base/<orgId>/<userId>/<sessionId>/<runId>/<stage>/<fileId>_<name>.<ext>
        """
        _, ext = os.path.splitext(original_filename)
        if not ext:
            ext = ".bin"
        safe_name = _sanitize(os.path.splitext(original_filename)[0])
        filename = f"{file_id}_{safe_name}{ext}"
        return "/".join([
            self.base_directory,
            org_id or "unknown_org",
            user_id or "unknown_user",
            session_id or "no_session",
            run_id or "no_run",
            stage,
            filename,
        ])

    def _run_prefix(self, org_id: str, user_id: str, session_id: str, run_id: str) -> str:
        """Prefix for all files belonging to a run."""
        return "/".join([
            self.base_directory,
            org_id or "unknown_org",
            user_id or "unknown_user",
            session_id or "no_session",
            run_id or "no_run",
        ]) + "/"

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def upload_file(
        self,
        file_data: bytes,
        original_filename: str,
        org_id: str = None,
        user_id: str = None,
        session_id: str = None,
        run_id: str = None,
        stage: str = "input",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Upload a file to Azure Blob Storage using hierarchical path.

        Args:
            file_data: Raw file bytes
            original_filename: Original filename
            org_id: Organization ID
            user_id: User ID
            session_id: Session / chat ID
            run_id: Workflow run ID
            stage: 'input', 'intermediate', or 'output'
            metadata: Optional extra metadata dict

        Returns:
            Dict with file_id, blob_path, blob_url, size_bytes, stage
        """
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {VALID_STAGES}")

        try:
            file_id = str(uuid.uuid4())
            blob_path = self._build_blob_path(
                org_id, user_id, session_id, run_id, stage, file_id, original_filename,
            )

            blob_metadata = metadata or {}
            blob_metadata.update({
                "original_filename": original_filename,
                "uploaded_at": datetime.utcnow().isoformat(),
                "org_id": org_id or "",
                "user_id": user_id or "",
                "session_id": session_id or "",
                "run_id": run_id or "",
                "stage": stage,
            })

            blob_client = self.container_client.get_blob_client(blob_path)
            blob_client.upload_blob(file_data, overwrite=True, metadata=blob_metadata)

            logger.info(f"📤 Uploaded: {original_filename} → {blob_path} ({len(file_data)} bytes) [{stage}]")

            return {
                "file_id": file_id,
                "blob_path": blob_path,
                "blob_url": blob_client.url,
                "size_bytes": len(file_data),
                "extension": os.path.splitext(original_filename)[1] or ".bin",
                "uploaded_at": blob_metadata["uploaded_at"],
                "stage": stage,
            }

        except AzureError as e:
            logger.error(f"❌ Azure blob upload failed for {original_filename}: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error during blob upload: {e}")
            raise

    def download_file(self, blob_path: str) -> bytes:
        """Download a file from Azure Blob Storage by its full blob path."""
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            download_stream = blob_client.download_blob()
            file_data = download_stream.readall()
            logger.info(f"📥 Downloaded: {blob_path} ({len(file_data)} bytes)")
            return file_data
        except ResourceNotFoundError:
            logger.error(f"❌ Blob not found: {blob_path}")
            raise FileNotFoundError(f"Blob not found: {blob_path}")
        except AzureError as e:
            logger.error(f"❌ Azure blob download failed for {blob_path}: {e}")
            raise

    def download_to_local(self, blob_path: str, local_dir: str = None) -> str:
        """Download blob to a local temp file and return the local path."""
        import tempfile
        file_data = self.download_file(blob_path)
        _, ext = os.path.splitext(blob_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, os.path.basename(blob_path))
            with open(local_path, "wb") as f:
                f.write(file_data)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(file_data)
            tmp.close()
            local_path = tmp.name
        return local_path

    def list_run_files(
        self,
        org_id: str,
        user_id: str,
        session_id: str,
        run_id: str,
        stage: str = None,
    ) -> List[Dict[str, Any]]:
        """List all files for a run, optionally filtered by stage."""
        try:
            prefix = self._run_prefix(org_id, user_id, session_id, run_id)
            if stage:
                prefix += f"{stage}/"
            blobs = self.container_client.list_blobs(name_starts_with=prefix, include=["metadata"])
            file_list = []
            for blob in blobs:
                file_list.append({
                    "blob_path": blob.name,
                    "size_bytes": blob.size,
                    "created_at": blob.creation_time.isoformat() if blob.creation_time else None,
                    "metadata": blob.metadata or {},
                })
            return file_list
        except AzureError as e:
            logger.error(f"❌ Azure blob list failed: {e}")
            return []

    def generate_sas_url(self, blob_path: str, expiry_hours: int = 24) -> str:
        """
        Generate a time-limited SAS URL for direct file access.
        
        Args:
            blob_path: Full blob path
            expiry_hours: Hours until URL expires (default: 24)
        
        Returns:
            SAS URL string
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=self.account_name,
                container_name=self.container_name,
                blob_name=blob_path,
                account_key=self.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
            )
            
            sas_url = f"{blob_client.url}?{sas_token}"
            logger.info(f"🔗 Generated SAS URL for {blob_path} (expires in {expiry_hours}h)")
            return sas_url

        except Exception as e:
            logger.error(f"❌ SAS URL generation failed: {e}")
            raise

    def blob_exists(self, blob_path: str) -> bool:
        """
        Check if a blob exists.
        
        Args:
            blob_path: Full blob path
        
        Returns:
            True if exists, False otherwise
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            return blob_client.exists()
        except Exception:
            return False
