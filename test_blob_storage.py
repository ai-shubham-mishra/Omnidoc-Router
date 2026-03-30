"""
Test script for Azure Blob Storage integration
Run this to verify blob storage is working correctly before deploying.
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import from Omnidoc-Router
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Omnidoc-Router'))

load_dotenv()

def test_blob_storage():
    """Test Azure Blob Storage operations."""
    print("=" * 80)
    print("Azure Blob Storage Test")
    print("=" * 80)
    
    # Check environment variables
    print("\n📋 Checking environment variables...")
    required_vars = [
        "AZ_BLOB_CONN_STRING",
        "AZ_BLOB_CONTAINER_NAME",
        "AZ_BLOB_STORAGE_ACCOUNT_NAME",
        "STORAGE_BACKEND"
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "CONN_STRING" in var or "KEY" in var:
                display = f"{value[:20]}...{value[-10:]}" if len(value) > 30 else value
            else:
                display = value
            print(f"  ✅ {var} = {display}")
        else:
            print(f"  ❌ {var} is not set!")
            return False
    
    # Test blob storage manager
    print("\n📦 Initializing BlobStorageManager...")
    try:
        from utils.blob_storage_manager import BlobStorageManager
        storage = BlobStorageManager()
        print("  ✅ BlobStorageManager initialized successfully")
    except Exception as e:
        print(f"  ❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test upload
    print("\n📤 Testing file upload...")
    test_data = b"Hello from OmniDoc Router! This is a test file."
    test_metadata = {
        "user_id": "test-user",
        "org_id": "test-org",
    }
    
    try:
        upload_result = storage.upload_file(
            service_name="router-test",
            session_or_run_id="test-session-123",
            file_data=test_data,
            original_filename="test-file.txt",
            metadata=test_metadata,
        )
        print(f"  ✅ Upload successful!")
        print(f"     File ID: {upload_result['file_id']}")
        print(f"     Blob path: {upload_result['blob_path']}")
        print(f"     Size: {upload_result['size_bytes']} bytes")
    except Exception as e:
        print(f"  ❌ Upload failed: {e}")
        return False
    
    # Test list
    print("\n📋 Testing file listing...")
    try:
        files = storage.list_session_files("router-test", "test-session-123")
        print(f"  ✅ Found {len(files)} file(s)")
        for f in files:
            print(f"     - {f['blob_path']} ({f['size_bytes']} bytes)")
    except Exception as e:
        print(f"  ❌ List failed: {e}")
        return False
    
    # Test download
    print("\n📥 Testing file download...")
    try:
        blob_path = upload_result['blob_path']
        downloaded_data = storage.download_file(blob_path)
        if downloaded_data == test_data:
            print(f"  ✅ Download successful! Data matches.")
        else:
            print(f"  ❌ Download successful but data mismatch!")
            return False
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return False
    
    # Test SAS URL generation
    print("\n🔗 Testing SAS URL generation...")
    try:
        sas_url = storage.generate_sas_url(blob_path, expiry_hours=1)
        print(f"  ✅ SAS URL generated (expires in 1h)")
        print(f"     URL: {sas_url[:80]}...")
    except Exception as e:
        print(f"  ❌ SAS URL generation failed: {e}")
        return False
    
    # Test cleanup
    print("\n🗑️ Testing cleanup...")
    try:
        deleted_count = storage.delete_session_files("router-test", "test-session-123")
        print(f"  ✅ Cleanup successful! Deleted {deleted_count} file(s)")
    except Exception as e:
        print(f"  ❌ Cleanup failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! Blob storage is ready.")
    print("=" * 80)
    return True


def test_storage_factory():
    """Test storage factory with backend selection."""
    print("\n" + "=" * 80)
    print("Storage Factory Test")
    print("=" * 80)
    
    print(f"\nCurrent STORAGE_BACKEND: {os.getenv('STORAGE_BACKEND', 'not set')}")
    
    try:
        from utils.storage_factory import get_storage_backend
        storage = get_storage_backend()
        print(f"✅ Storage backend initialized: {storage.__class__.__name__}")
        return True
    except Exception as e:
        print(f"❌ Storage factory failed: {e}")
        return False


if __name__ == "__main__":
    print("\n🧪 OmniDoc Router - Blob Storage Tests\n")
    
    # Test storage factory
    factory_ok = test_storage_factory()
    
    if not factory_ok:
        print("\n❌ Storage factory test failed! Fix configuration before proceeding.")
        sys.exit(1)
    
    # Test blob storage (only if backend is azure)
    if os.getenv("STORAGE_BACKEND", "azure").lower() == "azure":
        blob_ok = test_blob_storage()
        if not blob_ok:
            print("\n❌ Blob storage tests failed!")
            sys.exit(1)
    else:
        print("\n⚠️ Skipping blob storage tests (STORAGE_BACKEND != 'azure')")
    
    print("\n✅ All tests completed successfully!")
    print("Your Router is ready to use Azure Blob Storage! 🚀\n")
