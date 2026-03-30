"""
Centralized Azure Cosmos DB (Document DB) configuration for the Router.
All database connection and collection name references go through this module.

Cluster: omnidoc-dev-01
Database: omnidoc-backend-dev
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# ── Connection ──
AZ_COSMOS_DB_URL = os.getenv("AZ_COSMOS_DB_URL")
AZ_COSMOS_DB_NAME = os.getenv("AZ_COSMOS_DB_NAME")

# ── Collection Names (Router-relevant) ──
REGISTERED_WORKFLOWS_COLLECTION = os.getenv("AZ_COSMOS_DB_REGISTERED_WORKFLOWS_COLLECTION", "registeredWorkflows")
CHAT_SESSIONS_COLLECTION = os.getenv("AZ_COSMOS_DB_CHAT_SESSIONS_COLLECTION", "chat_sessions")


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
