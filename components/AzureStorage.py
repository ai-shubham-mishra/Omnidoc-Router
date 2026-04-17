"""
Azure Blob Storage Utilities for OmniDoc Router
Self-contained storage operations - uses Router's own KeyVaultClient.

Container: "knowledge-base" (with hyphen)
Directory structure inside container: {org_id}/{user_id}/{session_id}/{run_id}/{stage}/
Valid stages: input, intermediate, output

Local tmp structure: tmp/{session_id}/{run_id}/ or tmp/{run_id}/ if session_id not provided
"""

import os
import re
import uuid
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    generate_blob_sas,
    BlobSasPermissions,
)
from azure.core.exceptions import ResourceNotFoundError, AzureError
from dotenv import load_dotenv
from components.KeyVaultClient import get_secret

load_dotenv()
logger = logging.getLogger(__name__)

# ============== Configuration ==============
CONTAINER_NAME = get_secret("AZ_BLOB_CONTAINER_NAME", "knowledge-base")  # with hyphen
BASE_DIRECTORY = ""  # No prefix - files stored at container root
VALID_STAGES = ("input", "intermediate", "output")

# Azure credentials
CONN_STRING = get_secret("AZ_BLOB_CONN_STRING")
ACCOUNT_NAME = get_secret("AZ_BLOB_STORAGE_ACCOUNT_NAME", "omnidocdev")
ACCOUNT_KEY = get_secret("AZ_BLOB_CONN_KEY")

# Local tmp directory
TMP_BASE_DIR = os.path.join(os.getcwd(), "tmp")


# ============== Helper Functions ==============

def _sanitize(name: str, max_len: int = 120) -> str:
    """Remove unsafe characters from filename."""
    safe = re.sub(r"[^a-zA-Z0-9._\-]", "_", name)
    return safe[:max_len]


def _get_blob_service_client() -> BlobServiceClient:
    """Get Azure Blob Service Client."""
    if not CONN_STRING:
        raise ValueError("AZ_BLOB_CONN_STRING not found in environment variables")
    
    try:
        client = BlobServiceClient.from_connection_string(CONN_STRING)
        # Test connection
        client.get_container_client(CONTAINER_NAME).exists()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Azure Blob Storage: {e}")
        raise


def _build_blob_path(
    org_id: str,
    user_id: str,
    session_id: str,
    run_id: str,
    stage: str,
    filename: str,
) -> str:
    """
    Build hierarchical blob path.
    
    Two patterns:
    1. Session-level files (no run_id): {org}/{user}/{session}/{file}
    2. Workflow files (with run_id): {org}/{user}/{session}/{run}/{stage}/{file}
    """
    file_id = str(uuid.uuid4())[:8]
    safe_name = _sanitize(filename)
    
    # Pattern 1: Session-level files (router uploads)
    if not run_id or not stage:
        path_parts = [
            BASE_DIRECTORY,
            org_id or "unknown_org",
            user_id or "unknown_user",
            session_id or "no_session",
            f"{file_id}_{safe_name}",
        ]
        return "/".join(filter(None, path_parts))
    
    # Pattern 2: Workflow files (with run_id and stage)
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}")
    
    path_parts = [
        BASE_DIRECTORY,
        org_id or "unknown_org",
        user_id or "unknown_user",
        session_id or "no_session",
        run_id,
        stage,
        f"{file_id}_{safe_name}",
    ]
    return "/".join(filter(None, path_parts))


def get_tmp_path(run_id: str, session_id: str = None, filename: str = None) -> str:
    """
    Get local tmp path for downloaded files.
    Structure: tmp/{session_id}/{run_id}/ or tmp/{run_id}/ if no session
    """
    if session_id:
        base_path = os.path.join(TMP_BASE_DIR, _sanitize(session_id), _sanitize(run_id))
    else:
        base_path = os.path.join(TMP_BASE_DIR, _sanitize(run_id))
    
    os.makedirs(base_path, exist_ok=True)
    
    if filename:
        return os.path.join(base_path, _sanitize(filename))
    return base_path


# ============== Upload Functions ==============

def upload_file_to_blob(
    file_data: bytes,
    filename: str,
    org_id: str,
    user_id: str,
    run_id: Optional[str] = None,
    session_id: str = None,
    stage: Optional[str] = "input",
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Upload a single file to Azure Blob Storage.
    
    Args:
        file_data: File content as bytes
        filename: Original filename
        org_id: Organization ID
        user_id: User ID
        run_id: Optional workflow run ID (None for session-level files)
        session_id: Optional session ID
        stage: Optional file lifecycle stage (input/intermediate/output, None for session files)
        metadata: Optional metadata dict
    
    Returns:
        {
            "blob_path": "org/user/session/run/stage/file",
            "sas_url": "https://...",
            "local_tmp_path": "tmp/session/run/file.pdf",
            "file_id": "abc123",
            "original_filename": "file.pdf",
            "size_bytes": 12345
        }
    """
    try:
        # Build blob path (handles both session-level and workflow files)
        blob_path = _build_blob_path(org_id, user_id, session_id, run_id, stage, filename)
        
        # Upload to Azure
        blob_service_client = _get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_path)
        
        # Prepare metadata
        upload_metadata = {
            "original_filename": filename,
            "org_id": org_id,
            "user_id": user_id,
            "run_id": run_id or "",
            "session_id": session_id or "",
            "stage": stage or "",
            "uploaded_at": datetime.utcnow().isoformat(),
        }
        if metadata:
            upload_metadata.update(metadata)
        
        # Upload
        blob_client.upload_blob(file_data, overwrite=True, metadata=upload_metadata)
        
        # Generate SAS URL (24-hour expiry)
        sas_url = generate_sas_url(blob_path, expiry_hours=24)
        
        # Save to local tmp for immediate use
        # For session files (no run_id), use session_id as container
        local_tmp_path = get_tmp_path(run_id or session_id, session_id, filename)
        with open(local_tmp_path, 'wb') as f:
            f.write(file_data)
        
        logger.info(f"✅ Uploaded to blob: {blob_path} ({len(file_data)} bytes)")
        
        return {
            "blob_path": blob_path,
            "sas_url": sas_url,
            "local_tmp_path": local_tmp_path,
            "file_id": blob_path.split('/')[-1].split('_')[0],
            "original_filename": filename,
            "size_bytes": len(file_data),
        }
        
    except Exception as e:
        logger.error(f"Failed to upload {filename}: {e}")
        raise


async def upload_files_batch(
    files: List,  # List of FastAPI UploadFile objects
    org_id: str,
    user_id: str,
    run_id: str,
    session_id: str = None,
    stage: str = "input",
) -> Tuple[List[str], List[str]]:
    """
    Upload multiple files to blob storage.
    
    Returns:
        (blob_paths, local_tmp_paths)
    """
    blob_paths = []
    local_paths = []
    
    for file_obj in files:
        if not file_obj or not file_obj.filename:
            continue
        
        # Read file content
        content = await file_obj.read()
        
        # Upload
        result = upload_file_to_blob(
            file_data=content,
            filename=file_obj.filename,
            org_id=org_id,
            user_id=user_id,
            run_id=run_id,
            session_id=session_id,
            stage=stage,
        )
        
        blob_paths.append(result["blob_path"])
        local_paths.append(result["local_tmp_path"])
    
    return blob_paths, local_paths


# ============== Download Functions ==============

def download_from_blob_path(
    blob_path: str,
    run_id: str,
    session_id: str = None,
) -> str:
    """
    Download file from blob storage by blob path.
    Saves to tmp/{session_id}/{run_id}/ or tmp/{run_id}/
    
    Returns:
        Local file path
    """
    try:
        # Extract filename from blob path
        filename = blob_path.split('/')[-1]
        # Remove file_id prefix (abc123_filename.pdf -> filename.pdf)
        if '_' in filename:
            filename = '_'.join(filename.split('_')[1:])
        
        # Get local tmp path
        local_path = get_tmp_path(run_id, session_id, filename)
        
        # Download from blob
        blob_service_client = _get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_path)
        
        with open(local_path, 'wb') as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
        
        logger.info(f"✅ Downloaded from blob: {blob_path} -> {local_path}")
        return local_path
        
    except ResourceNotFoundError:
        logger.error(f"Blob not found: {blob_path}")
        raise FileNotFoundError(f"Blob not found: {blob_path}")
    except Exception as e:
        logger.error(f"Failed to download {blob_path}: {e}")
        raise


def download_from_sas_url(
    sas_url: str,
    run_id: str,
    session_id: str = None,
    filename: str = None,
) -> str:
    """
    Download file from Azure Blob Storage using SAS URL.
    
    Args:
        sas_url: Azure Blob SAS URL
        run_id: Workflow run ID
        session_id: Optional session ID
        filename: Optional filename override (otherwise extracted from URL)
    
    Returns:
        Local file path
    """
    import requests
    
    try:
        # Extract filename from URL if not provided
        if not filename:
            # Parse URL path: .../org/user/session/run/stage/abc123_file.pdf?sas_token
            url_path = sas_url.split('?')[0]  # Remove SAS token
            filename = url_path.split('/')[-1]
            # Remove file_id prefix
            if '_' in filename:
                filename = '_'.join(filename.split('_')[1:])
        
        # Get local tmp path
        local_path = get_tmp_path(run_id, session_id, filename)
        
        # Download from SAS URL
        response = requests.get(sas_url, timeout=60)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"✅ Downloaded from SAS URL -> {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"Failed to download from SAS URL: {e}")
        raise


# ============== SAS URL Generation ==============

def generate_sas_url(blob_path: str, expiry_hours: int = 24) -> str:
    """
    Generate SAS URL for blob with time-limited access.
    
    Args:
        blob_path: Blob path in container
        expiry_hours: SAS URL validity period (default 24 hours)
    
    Returns:
        SAS URL string
    """
    try:
        if not ACCOUNT_NAME or not ACCOUNT_KEY:
            raise ValueError("Azure account name and key required for SAS generation")
        
        # Generate SAS token
        sas_token = generate_blob_sas(
            account_name=ACCOUNT_NAME,
            container_name=CONTAINER_NAME,
            blob_name=blob_path,
            account_key=ACCOUNT_KEY,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
        )
        
        # Construct full URL
        sas_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/{blob_path}?{sas_token}"
        
        return sas_url
        
    except Exception as e:
        logger.error(f"Failed to generate SAS URL for {blob_path}: {e}")
        raise


# ============== List & Delete Functions ==============

def list_run_files(
    org_id: str,
    user_id: str,
    run_id: str,
    session_id: str = None,
    stage: str = None,
) -> List[Dict[str, Any]]:
    """
    List all files for a workflow run.
    
    Returns:
        List of file metadata dicts
    """
    try:
        # Build prefix
        prefix = "/".join([
            BASE_DIRECTORY,
            org_id or "unknown_org",
            user_id or "unknown_user",
            session_id or "no_session",
            run_id or "no_run",
        ])
        
        if stage:
            prefix = f"{prefix}/{stage}/"
        else:
            prefix = f"{prefix}/"
        
        # List blobs
        blob_service_client = _get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        files = []
        for blob in container_client.list_blobs(name_starts_with=prefix):
            files.append({
                "blob_path": blob.name,
                "filename": blob.name.split('/')[-1],
                "size_bytes": blob.size,
                "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                "metadata": blob.metadata,
            })
        
        logger.info(f"Found {len(files)} files for run {run_id}")
        return files
        
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise


def delete_run_files(
    org_id: str,
    user_id: str,
    run_id: str,
    session_id: str = None,
) -> int:
    """
    Delete all files for a workflow run.
    
    Returns:
        Number of files deleted
    """
    try:
        # List all files
        files = list_run_files(org_id, user_id, run_id, session_id)
        
        blob_service_client = _get_blob_service_client()
        
        deleted_count = 0
        for file_info in files:
            blob_client = blob_service_client.get_blob_client(
                container=CONTAINER_NAME,
                blob=file_info["blob_path"]
            )
            blob_client.delete_blob()
            deleted_count += 1
        
        logger.info(f"Deleted {deleted_count} files for run {run_id}")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Failed to delete files: {e}")
        raise


# ============== Local Tmp Cleanup ==============

def cleanup_tmp_directory(run_id: str, session_id: str = None) -> bool:
    """
    Delete local tmp files for a workflow run.
    
    Returns:
        True if successful
    """
    try:
        tmp_path = get_tmp_path(run_id, session_id)
        
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
            logger.info(f"Cleaned up tmp directory: {tmp_path}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to cleanup tmp directory: {e}")
        return False


def cleanup_all_tmp() -> int:
    """
    Cleanup all tmp directories older than 24 hours.
    
    Returns:
        Number of directories cleaned
    """
    try:
        if not os.path.exists(TMP_BASE_DIR):
            return 0
        
        cleaned = 0
        cutoff_time = datetime.now().timestamp() - (24 * 3600)  # 24 hours ago
        
        for root, dirs, files in os.walk(TMP_BASE_DIR):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                # Check modification time
                if os.path.getmtime(dir_path) < cutoff_time:
                    shutil.rmtree(dir_path)
                    cleaned += 1
        
        logger.info(f"Cleaned up {cleaned} old tmp directories")
        return cleaned
        
    except Exception as e:
        logger.error(f"Failed to cleanup old tmp directories: {e}")
        return 0
