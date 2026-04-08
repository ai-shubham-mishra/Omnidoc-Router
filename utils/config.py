import os
from dotenv import load_dotenv

load_dotenv()

API_VERSION = "2.0.0"
LAST_UPDATED = "2026-03-19"

# Session TTL (3 days in seconds)
SESSION_TTL_SECONDS = 259200  # 3 days

# Storage Backend Selection ('azure' or 'local')
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "azure")

# Hierarchical blob base directory
AZ_BLOB_BASE_DIRECTORY = os.getenv("AZ_BLOB_BASE_DIRECTORY", "knowledge_base")

# Note: Azure Blob credentials are fetched directly by BlobStorageManager
# from Azure Key Vault using components/KeyVaultClient.py
