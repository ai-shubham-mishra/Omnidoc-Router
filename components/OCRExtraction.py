"""
OCR Extraction Component for Router - Simple text extraction for file classification.

Provides lightweight OCR extraction using Google's Gemini API for file classification purposes.
This is a Router-specific implementation, separate from AgenticAPI's full OCR capabilities.

Note: google.generativeai is imported lazily to avoid requiring it when not in use.
"""

import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def extract_text_from_file(
    file_id: str, 
    max_chars: int = 2000,
    use_cache: bool = True
) -> str:
    """
    Extract raw text from file for classification purposes.
    Uses Gemini to extract text from first page only for speed.
    
    This function is used by FileClassifier for OCR-based file matching.
    Results are cached in file metadata to avoid redundant processing.
    
    Args:
        file_id: File ID to extract text from
        max_chars: Maximum characters to extract (default 2000 for first page)
        use_cache: Whether to check/store cache in file metadata (default True)
    
    Returns:
        Extracted text string (up to max_chars), or empty string on failure
    """
    try:
        # Check cache first if enabled
        if use_cache:
            cached_text = await _get_cached_ocr(file_id)
            if cached_text:
                logger.debug(f"Using cached OCR for file {file_id}")
                return cached_text[:max_chars]
        
        # Import dependencies lazily
        import google.generativeai as genai
        from components.FileMiddleware import file_middleware
        from components.KeyVaultClient import get_secret
        
        # Get Gemini API key from Key Vault
        gemini_api_key = get_secret("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY not found in Key Vault")
            return ""
        
        genai.configure(api_key=gemini_api_key)
        
        # Download file from Azure Blob
        file_path = await file_middleware.download_file(file_id, as_bytes=False)
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File not found: {file_id}")
            return ""
        
        # Upload to Gemini for processing
        uploaded_file = genai.upload_file(file_path)
        
        # Wait for Gemini to process the file (max 30 seconds)
        max_wait = 30
        elapsed = 0
        while uploaded_file.state.name == "PROCESSING" and elapsed < max_wait:
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
            elapsed += 2
        
        if uploaded_file.state.name != "ACTIVE":
            logger.warning(f"Gemini file processing failed for {file_id}")
            genai.delete_file(uploaded_file.name)
            return ""
        
        # Extract text from first page only (fast classification)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Extract all visible text from this document. "
            f"Return only the text content, no formatting or markdown. "
            f"Limit to approximately the first {max_chars} characters."
        )
        
        response = model.generate_content([uploaded_file, prompt])
        
        # Cleanup Gemini storage
        genai.delete_file(uploaded_file.name)
        
        # Extract and truncate text
        text = response.text if hasattr(response, 'text') else ""
        text = text[:max_chars]
        
        # Cache result if enabled
        if use_cache and text:
            await _cache_ocr(file_id, text)
        
        logger.info(f"Extracted {len(text)} chars from file {file_id}")
        return text
        
    except ImportError as e:
        logger.error(f"Missing dependency for OCR: {e}")
        return ""
    except Exception as e:
        logger.error(f"Text extraction failed for {file_id}: {e}")
        return ""


async def _get_cached_ocr(file_id: str) -> Optional[str]:
    """
    Retrieve cached OCR result from file metadata.
    
    Args:
        file_id: File ID to check
    
    Returns:
        Cached OCR text or None if not cached
    """
    try:
        from utils.db_config import get_database
        
        db = get_database()
        files_collection = db["files"]
        
        # PyMongo is synchronous, no await needed
        file_doc = files_collection.find_one(
            {"_id": file_id},
            {"metadata.ocr_cache": 1}
        )
        
        if file_doc and "metadata" in file_doc and "ocr_cache" in file_doc["metadata"]:
            return file_doc["metadata"]["ocr_cache"]
        
        return None
        
    except Exception as e:
        logger.debug(f"Cache retrieval failed for {file_id}: {e}")
        return None


async def _cache_ocr(file_id: str, text: str) -> bool:
    """
    Store OCR result in file metadata for future use.
    
    Args:
        file_id: File ID to cache for
        text: Extracted text to cache
    
    Returns:
        True if cached successfully, False otherwise
    """
    try:
        from utils.db_config import get_database
        from datetime import datetime
        
        db = get_database()
        files_collection = db["files"]
        
        # PyMongo is synchronous, no await needed
        result = files_collection.update_one(
            {"_id": file_id},
            {
                "$set": {
                    "metadata.ocr_cache": text,
                    "metadata.ocr_cached_at": datetime.utcnow()
                }
            }
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        logger.debug(f"Cache storage failed for {file_id}: {e}")
        return False


def clear_ocr_cache(file_id: str) -> bool:
    """
    Clear cached OCR result for a file.
    Useful if file content changes or cache becomes invalid.
    
    Args:
        file_id: File ID to clear cache for
    
    Returns:
        True if cleared successfully, False otherwise
    """
    try:
        from utils.db_config import get_database
        
        db = get_database()
        files_collection = db["files"]
        
        result = files_collection.update_one(
            {"_id": file_id},
            {
                "$unset": {
                    "metadata.ocr_cache": "",
                    "metadata.ocr_cached_at": ""
                }
            }
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"Cache clearing failed for {file_id}: {e}")
        return False
