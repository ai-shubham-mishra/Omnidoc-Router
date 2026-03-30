import os
from dotenv import load_dotenv

load_dotenv()

API_VERSION = "2.0.0"
LAST_UPDATED = "2026-03-19"

# Session TTL (3 days in seconds)
SESSION_TTL_SECONDS = 259200  # 3 days

# Azure Blob Storage Configuration
AZ_BLOB_CONN_STRING = os.getenv("AZ_BLOB_CONN_STRING")
AZ_BLOB_STORAGE_ACCOUNT_NAME = os.getenv("AZ_BLOB_STORAGE_ACCOUNT_NAME")
AZ_BLOB_CONN_KEY = os.getenv("AZ_BLOB_CONN_KEY")
AZ_BLOB_CONTAINER_NAME = os.getenv("AZ_BLOB_CONTAINER_NAME")

# Storage Backend Selection ('azure' or 'local')
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "azure")

# Hierarchical blob base directory
AZ_BLOB_BASE_DIRECTORY = os.getenv("AZ_BLOB_BASE_DIRECTORY", "knowledge-base")
