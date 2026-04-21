"""
Pydantic schemas for structured LLM responses.
Used with Azure OpenAI json_schema mode to ensure consistent, well-formatted outputs.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class TableRow(BaseModel):
    """A single row in a table."""
    cells: List[str] = Field(..., description="Cell values for this row")


class Table(BaseModel):
    """A markdown table."""
    headers: List[str] = Field(..., description="Column headers")
    rows: List[TableRow] = Field(..., description="Table rows")
    caption: Optional[str] = Field(None, description="Optional table caption/title")


class BulletList(BaseModel):
    """A bullet list with optional nested items."""
    items: List[str] = Field(..., description="List items")


class CodeBlock(BaseModel):
    """A code block with language specification."""
    language: str = Field(..., description="Programming language (e.g., python, json)")
    code: str = Field(..., description="Code content")


class ContentSection(BaseModel):
    """A section of content with specific type."""
    type: Literal["heading", "paragraph", "table", "bullets", "code"] = Field(
        ..., description="Type of content section"
    )
    level: Optional[int] = Field(None, description="Heading level (1-6) if type is heading")
    text: Optional[str] = Field(None, description="Text content for heading/paragraph")
    table: Optional[Table] = Field(None, description="Table data if type is table")
    bullets: Optional[BulletList] = Field(None, description="Bullet list if type is bullets")
    code: Optional[CodeBlock] = Field(None, description="Code block if type is code")


class StructuredResponse(BaseModel):
    """
    Structured response schema for Azure OpenAI json_schema mode.
    The LLM returns this structure, and we build perfect markdown from it.
    """
    sections: List[ContentSection] = Field(
        ..., 
        description="Ordered list of content sections that make up the response"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "sections": [
                    {
                        "type": "heading",
                        "level": 2,
                        "text": "Analysis Results"
                    },
                    {
                        "type": "paragraph",
                        "text": "The workflow completed successfully with the following insights:"
                    },
                    {
                        "type": "bullets",
                        "bullets": {
                            "items": [
                                "Channel A delivered 124% ROI",
                                "Campaign B exceeded targets by 15%"
                            ]
                        }
                    },
                    {
                        "type": "table",
                        "table": {
                            "caption": "Performance Metrics",
                            "headers": ["Metric", "Value", "Status"],
                            "rows": [
                                {"cells": ["Revenue", "$50,000", "✅ On target"]},
                                {"cells": ["Conversion", "3.2%", "⚠️ Below target"]}
                            ]
                        }
                    }
                ]
            }
        }
