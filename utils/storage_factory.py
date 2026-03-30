"""
Storage Factory for OmniDoc Router
Provides environment-driven storage backend selection (Azure Blob vs Local tmp/).
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_storage_backend():
    """
    Get the appropriate storage backend based on STORAGE_BACKEND environment variable.
    
    Returns:
        BlobStorageManager if STORAGE_BACKEND='azure'
        LocalStorageManager if STORAGE_BACKEND='local' (for dev/testing)
    
    Default: azure
    """
    backend_type = os.getenv("STORAGE_BACKEND", "azure").lower()
    
    if backend_type == "azure":
        try:
            from .blob_storage_manager import BlobStorageManager
            logger.info("📦 Using Azure Blob Storage backend")
            return BlobStorageManager()
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure Blob Storage: {e}")
            logger.warning("⚠️ Falling back to local storage")
            # Fallback to local if Azure fails
            from .local_storage_manager import LocalStorageManager
            return LocalStorageManager()
    
    elif backend_type == "local":
        from .local_storage_manager import LocalStorageManager
        logger.info("📁 Using Local (tmp/) storage backend")
        return LocalStorageManager()
    
    else:
        logger.warning(f"⚠️ Unknown STORAGE_BACKEND '{backend_type}', defaulting to Azure")
        from .blob_storage_manager import BlobStorageManager
        return BlobStorageManager()
