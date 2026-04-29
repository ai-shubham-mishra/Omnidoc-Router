"""
Centralized Azure Cosmos DB (Document DB) configuration for the Router.
All database connection and collection name references go through this module.
Fetches secrets from Azure Key Vault.

Cluster: omnidoc-dev-01
Database: omnidoc-backend-dev
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from components.KeyVaultClient import get_secret

load_dotenv()

# ── Connection ──
# Fetch from Key Vault (fallback to env for local dev without vault)
AZ_COSMOS_DB_URL = get_secret("AZ_COSMOS_DB_URL", default=os.getenv("AZ_COSMOS_DB_URL"))
AZ_COSMOS_DB_NAME = get_secret("AZ_COSMOS_DB_NAME", default=os.getenv("AZ_COSMOS_DB_NAME", "omnidoc-backend-dev"))

# ── Collection Names (Router-relevant) ──
REGISTERED_WORKFLOWS_COLLECTION = get_secret("AZ_COSMOS_DB_REGISTERED_WORKFLOWS_COLLECTION", default="registeredWorkflows")
CHAT_SESSIONS_COLLECTION = get_secret("AZ_COSMOS_DB_CHAT_SESSIONS_COLLECTION", default="chat_sessions")
IDLE_WORKFLOWS_COLLECTION = "idle_workflows"  # New: For workflows awaiting files


def get_db_client():
    """Create and return a MongoClient connected to Azure Cosmos DB."""
    return MongoClient(AZ_COSMOS_DB_URL)


def get_database():
    """Get the database instance."""
    client = get_db_client()
    return client[AZ_COSMOS_DB_NAME]


def get_collection(collection_name: str):
    """Get a specific collection from the database."""
    db = get_database()
    return db[collection_name]
