"""
Azure Key Vault Client for Omnidoc Router.
Fetches secrets (like Pinecone API key) from Azure Key Vault.
"""
import os
import logging
from typing import Optional
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.core.exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)


class KeyVaultClient:
    """Singleton client for Azure Key Vault access."""
    
    _instance = None
    _client: Optional[SecretClient] = None
    _cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the Key Vault client."""
        vault_url = os.getenv("AZURE_KEY_VAULT_URL")
        
        if not vault_url:
            logger.warning("AZURE_KEY_VAULT_URL not set - Key Vault features disabled")
            self._client = None
            return
        
        try:
            tenant_id = os.getenv("AZURE_TENANT_ID")
            client_id = os.getenv("AZURE_CLIENT_ID")
            client_secret = os.getenv("AZURE_CLIENT_SECRET")
            
            if tenant_id and client_id and client_secret:
                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
                logger.info("Using Service Principal for Key Vault")
            else:
                credential = DefaultAzureCredential()
                logger.info("Using Managed Identity for Key Vault")
            
            self._client = SecretClient(vault_url=vault_url, credential=credential)
            logger.info(f"✅ Key Vault client initialized: {vault_url}")
        
        except Exception as e:
            logger.error(f"❌ Key Vault initialization failed: {e}")
            self._client = None
    
    def get_secret(self, secret_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get secret from Azure Key Vault with caching.
        Converts underscores to hyphens for Azure compatibility.
        """
        if not self._client:
            logger.warning(f"Key Vault unavailable - returning default for {secret_name}")
            return default
        
        # Convert underscore to hyphen for Azure Key Vault
        vault_name = secret_name.replace("_", "-")
        
        # Check cache
        if vault_name in self._cache:
            return self._cache[vault_name]
        
        try:
            secret = self._client.get_secret(vault_name)
            self._cache[vault_name] = secret.value
            logger.debug(f"✅ Retrieved secret: {vault_name}")
            return secret.value
        
        except ResourceNotFoundError:
            logger.warning(f"Secret not found: {vault_name}")
            return default
        
        except Exception as e:
            logger.error(f"Error fetching secret {vault_name}: {e}")
            return default


# Global singleton instance
_kv_client = None

def get_secret(secret_name: str, default: Optional[str] = None) -> Optional[str]:
    """Global function to get secrets from Key Vault."""
    global _kv_client
    if _kv_client is None:
        _kv_client = KeyVaultClient()
    return _kv_client.get_secret(secret_name, default)
