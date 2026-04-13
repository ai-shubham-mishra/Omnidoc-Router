"""
File Storage Middleware - Bidirectional Conversion Layer (Router-specific)
Handles file upload/download with file_id tracking for workflow reusability.

Key Features:
- Upload binary files → Generate file_ids + store metadata in MongoDB
- Validate file_ids for user/org access
- Convert file_ids → Binary files for workflow execution
- Minimal metadata (user_id, org_id, session_id only)
"""

import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import mimetypes

from components.AzureStorage import upload_file_to_blob, download_from_blob_path, generate_sas_url
from utils.db_config import get_database

logger = logging.getLogger(__name__)


class FileMiddleware:
    """Middleware for file_id based workflow communication."""
    
    def __init__(self):
        self.db = get_database()
        self.files_collection = self.db["files"]
        logger.info("📦 FileMiddleware initialized")
    
    
    async def upload_files(
        self,
        files: List,
        user_id: str,
        org_id: str,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Upload binary files to Azure Blob, store metadata in MongoDB, return file_ids.
        
        Args:
            files: List of FastAPI UploadFile objects
            user_id: User ID from JWT
            org_id: Organization ID from JWT
            session_id: Optional session ID
            run_id: Optional run ID (None for router session files, UUID for workflow files)
            stage: Optional file stage (None for router, "input"/"intermediate"/"output" for workflows)
        
        Returns:
            List of file metadata dicts with file_id
        """
        uploaded_files = []
        # DO NOT auto-generate run_id for router uploads
        # run_id should only be set when files are sent to workflows
        
        for file in files:
            if not file or not file.filename:
                continue
            
            try:
                # Read file content (async for FastAPI UploadFile)
                content = await file.read()
                
                # Generate unique file_id
                file_id = str(uuid.uuid4())
                
                # Detect MIME type
                mime_type, _ = mimetypes.guess_type(file.filename)
                if not mime_type:
                    mime_type = file.content_type or "application/octet-stream"
                
                # Upload to Azure Blob (synchronous)
                # If no run_id, files go to session root
                # If run_id provided, files go to run_id/stage structure
                upload_result = upload_file_to_blob(
                    file_data=content,
                    filename=file.filename,
                    org_id=org_id,
                    user_id=user_id,
                    run_id=run_id,  # None for router uploads, UUID for workflow files
                    session_id=session_id,
                    stage=stage if run_id else None,  # Only use stage if part of workflow
                    metadata={"file_id": file_id}
                )
                
                # Store metadata in MongoDB (synchronous - no await)
                file_metadata = {
                    "_id": file_id,
                    "file_id": file_id,
                    "original_name": file.filename,
                    "blob_path": upload_result["blob_path"],
                    "blob_url": upload_result["sas_url"],
                    "mime_type": mime_type,
                    "size_bytes": len(content),
                    "user_id": user_id,
                    "org_id": org_id,
                    "session_id": session_id,
                    "run_id": run_id,
                    "stage": stage,
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "expires_at": (datetime.utcnow() + timedelta(days=90)).isoformat()
                }
                
                self.files_collection.insert_one(file_metadata)
                
                uploaded_files.append({
                    "file_id": file_id,
                    "original_name": file.filename,
                    "mime_type": mime_type,
                    "size_bytes": len(content),
                    "blob_path": upload_result["blob_path"]
                })
                
                logger.info(f"✅ File uploaded: {file.filename} → file_id: {file_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to upload {file.filename}: {e}")
                continue
        
        return uploaded_files
    
    
    def validate_file_ids(
        self,
        file_ids: List[str],
        user_id: str,
        org_id: str
    ) -> List[str]:
        """
        Validate that file_ids exist and belong to user/org.
        
        Args:
            file_ids: List of file IDs to validate
            user_id: User ID from JWT
            org_id: Organization ID from JWT
        
        Returns:
            List of valid file_ids
        """
        valid_ids = []
        
        for file_id in file_ids:
            try:
                file_doc = self.files_collection.find_one({
                    "file_id": file_id,
                    "$or": [
                        {"user_id": user_id},
                        {"org_id": org_id}
                    ]
                })
                
                if file_doc:
                    # Check expiry
                    expires_at = datetime.fromisoformat(file_doc["expires_at"])
                    if expires_at > datetime.utcnow():
                        valid_ids.append(file_id)
                    else:
                        logger.warning(f"File {file_id} expired at {expires_at}")
                else:
                    logger.warning(f"File {file_id} not found or access denied")
                    
            except Exception as e:
                logger.error(f"Error validating file_id {file_id}: {e}")
                continue
        
        return valid_ids
    
    
    def get_files_metadata(
        self,
        file_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Fetch metadata for given file_ids.
        
        Args:
            file_ids: List of file IDs
        
        Returns:
            List of file metadata dicts
        """
        cursor = self.files_collection.find({"file_id": {"$in": file_ids}})
        files = list(cursor)
        
        # Remove MongoDB _id field from response
        for file_doc in files:
            file_doc.pop("_id", None)
        
        return files
    
    
    def download_file(
        self,
        file_id: str,
        as_bytes: bool = False
    ) -> Union[str, bytes]:
        """
        Download file from blob storage by file_id.
        
        Args:
            file_id: The file identifier
            as_bytes: If True, return bytes. If False, return temp file path.
        
        Returns:
            File path (str) or file bytes (bytes)
        """
        # Get metadata
        file_doc = self.files_collection.find_one({"file_id": file_id})
        
        if not file_doc:
            raise FileNotFoundError(f"File {file_id} not found")
        
        # Download from blob using existing Azure utility
        blob_path = file_doc["blob_path"]
        run_id = file_doc.get("run_id")  # May be None for session-level files
        session_id = file_doc.get("session_id")
        
        # For tmp path, use run_id if available, otherwise session_id
        local_path = download_from_blob_path(
            blob_path=blob_path,
            run_id=run_id or session_id,  # Fallback to session_id for session-level files
            session_id=session_id
        )
        
        if as_bytes:
            with open(local_path, 'rb') as f:
                return f.read()
        
        return local_path
    
    
    def files_to_multipart(
        self,
        file_ids: List[str],
        field_name: str = "files"
    ) -> List[Tuple[str, Tuple[str, bytes]]]:
        """
        Convert file_ids to multipart form-data format for workflow APIs.
        Workflows receive binary files as before - no knowledge of file_ids.
        
        Args:
            file_ids: List of file identifiers
            field_name: Form field name (e.g., "files", "po_document")
        
        Returns:
            List of (field_name, (filename, file_bytes)) tuples
        """
        multipart_files = []
        
        for file_id in file_ids:
            try:
                file_doc = self.files_collection.find_one({"file_id": file_id})
                
                if not file_doc:
                    logger.warning(f"File {file_id} not found, skipping")
                    continue
                
                # Download file as bytes
                file_bytes = self.download_file(file_id, as_bytes=True)
                
                # Add to multipart list
                multipart_files.append(
                    (field_name, (file_doc["original_name"], file_bytes))
                )
                
                logger.info(f"Converted file_id {file_id} → binary for workflow")
                
            except Exception as e:
                logger.error(f"Failed to convert file_id {file_id}: {e}")
                continue
        
        return multipart_files
    
    
    def get_file_metadata_by_id(
        self,
        file_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get single file metadata by ID."""
        file_doc = self.files_collection.find_one({"file_id": file_id})
        
        if file_doc:
            file_doc.pop("_id", None)
        
        return file_doc
    
    
    def cleanup_expired_files(self) -> int:
        """
        Cleanup expired files from database and blob storage.
        Should be run periodically via cron/scheduler.
        
        Returns:
            Number of files cleaned up
        """
        try:
            # Find expired files
            expired_files = list(self.files_collection.find({
                "expires_at": {"$lt": datetime.utcnow().isoformat()}
            }))
            
            cleaned_count = 0
            
            for file_doc in expired_files:
                try:
                    # TODO: Delete from blob storage
                    # (Azure cleanup can be done separately)
                    
                    # Delete from database
                    self.files_collection.delete_one({"file_id": file_doc["file_id"]})
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to cleanup file {file_doc['file_id']}: {e}")
                    continue
            
            logger.info(f"🧹 Cleaned up {cleaned_count} expired files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error in cleanup_expired_files: {e}")
            return 0


# Singleton instance
file_middleware = FileMiddleware()
