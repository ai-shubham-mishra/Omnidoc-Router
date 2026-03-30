"""
Diagnostic script to check Azure Storage Account configuration
Run this to verify your storage account supports the features we need.
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Omnidoc-Router'))
load_dotenv()

def check_storage_account():
    """Check storage account configuration and capabilities."""
    print("=" * 80)
    print("Azure Storage Account Diagnostics")
    print("=" * 80)
    
    # Get connection string
    conn_string = os.getenv("AZ_BLOB_CONN_STRING")
    container_name = os.getenv("AZ_BLOB_CONTAINER_NAME")
    
    if not conn_string:
        print("\n❌ AZ_BLOB_CONN_STRING not found in environment!")
        return False
    
    try:
        from azure.storage.blob import BlobServiceClient
        
        print("\n🔌 Connecting to storage account...")
        blob_service = BlobServiceClient.from_connection_string(conn_string)
        
        # Get account information
        print("\n📊 Storage Account Information:")
        try:
            account_info = blob_service.get_account_information()
            print(f"  Account Kind: {account_info.get('account_kind', 'Unknown')}")
            print(f"  SKU Name: {account_info.get('sku_name', 'Unknown')}")
            
            # Check if Premium
            sku = str(account_info.get('sku_name', ''))
            if 'Premium' in sku:
                print("\n  ❌ ISSUE DETECTED: Premium Storage Account")
                print("     Premium accounts typically only support Page Blobs and Append Blobs.")
                print("     Block Blobs (needed for general file storage) are NOT supported.")
                print("\n  🔧 SOLUTION:")
                print("     1. Create a NEW storage account in Azure Portal")
                print("     2. Choose 'Standard' performance tier (NOT Premium)")
                print("     3. Choose 'StorageV2 (general purpose v2)' account kind")
                print("     4. Update AZ_BLOB_CONN_STRING in .env with new account")
                return False
            else:
                print("  ✅ Standard storage account detected (supports block blobs)")
        except Exception as e:
            print(f"  ⚠️ Could not get account info: {e}")
            print("     (This is often normal for Cosmos DB for MongoDB accounts)")
            print("     Proceeding with other checks...")
        
        # Check container
        if container_name:
            print(f"\n📦 Container: {container_name}")
            try:
                container_client = blob_service.get_container_client(container_name)
                props = container_client.get_container_properties()
                print(f"  ✅ Container exists and is accessible")
                print(f"     Created: {props.get('last_modified', 'Unknown')}")
            except Exception as e:
                print(f"  ❌ Container issue: {e}")
                return False
        
        # Check service properties
        print("\n⚙️ Service Properties:")
        try:
            service_props = blob_service.get_service_properties()
            print(f"  ✅ Service properties accessible")
        except Exception as e:
            print(f"  ⚠️ Could not get service properties (this is usually fine): {e.__class__.__name__}")
        
        print("\n✅ Diagnostics complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🔍 Running Azure Storage Account Diagnostics...\n")
    
    success = check_storage_account()
    
    if success:
        print("\n✅ Your storage account looks good!")
        print("   You can proceed with testing blob storage.\n")
    else:
        print("\n❌ Storage account configuration issues detected!")
        print("   Please review the recommendations above.\n")
        sys.exit(1)
