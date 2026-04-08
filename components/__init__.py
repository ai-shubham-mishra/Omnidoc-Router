"""Components module for Omnidoc Router."""

from .KeyVaultClient import get_secret, KeyVaultClient
from .PineconeClient import pinecone_client, PineconeClient
from .TextExtractor import text_extractor, TextExtractor

__all__ = [
    "get_secret",
    "KeyVaultClient",
    "pinecone_client",
    "PineconeClient",
    "text_extractor",
    "TextExtractor",
]
