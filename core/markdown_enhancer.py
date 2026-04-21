"""
Markdown builder for the LLM Router.
Builds perfect, structurally valid markdown from structured LLM responses
using Azure OpenAI json_schema mode. Guarantees consistent formatting.
"""
import logging
from typing import Optional, Dict, Any

from models.response_schemas import StructuredResponse, ContentSection, Table

logger = logging.getLogger(__name__)


class MarkdownEnhancer:
    """
    Builds perfect markdown from structured LLM responses.
    Takes a StructuredResponse (from json_schema mode) and generates
    flawless markdown with proper tables, bullets, headers, and spacing.
    """

    def build_from_structured(
        self, 
        structured_response: StructuredResponse,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build markdown from a StructuredResponse object.
        Returns clean, well-formatted markdown string.
        """
        if not structured_response or not structured_response.sections:
            return ""
        
        try:
            parts = []
            for section in structured_response.sections:
                markdown_chunk = self._render_section(section)
                if markdown_chunk:
                    parts.append(markdown_chunk)
            
            # Join with proper spacing
            return "\n\n".join(parts).strip()
        
        except Exception as e:
            logger.error(f"Failed to build markdown from structured response: {e}")
            return ""

    def _render_section(self, section: ContentSection) -> str:
        """Render a single content section to markdown."""
        section_type = section.type
        
        if section_type == "heading":
            return self._render_heading(section.level, section.text)
        
        elif section_type == "paragraph":
            return section.text or ""
        
        elif section_type == "table":
            return self._render_table(section.table)
        
        elif section_type == "bullets":
            return self._render_bullets(section.bullets.items if section.bullets else [])
        
        elif section_type == "code":
            return self._render_code(
                section.code.language if section.code else "text",
                section.code.code if section.code else ""
            )
        
        return ""

    def _render_heading(self, level: Optional[int], text: Optional[str]) -> str:
        """Render a markdown heading."""
        if not text:
            return ""
        level = max(1, min(6, level or 2))  # Clamp to 1-6
        return f"{'#' * level} {text}"

    def _render_table(self, table: Optional[Table]) -> str:
        """
        Render a perfectly aligned markdown table.
        Calculates column widths for proper alignment.
        """
        if not table or not table.headers or not table.rows:
            return ""
        
        lines = []
        
        # Add caption if present
        if table.caption:
            lines.append(f"**{table.caption}**")
            lines.append("")  # Blank line after caption
        
        # Calculate column widths for alignment
        num_cols = len(table.headers)
        col_widths = [len(h) for h in table.headers]
        
        for row in table.rows:
            for i, cell in enumerate(row.cells[:num_cols]):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Build header row
        header_cells = [
            h.ljust(col_widths[i]) 
            for i, h in enumerate(table.headers)
        ]
        lines.append("| " + " | ".join(header_cells) + " |")
        
        # Build separator row
        separator_cells = ["-" * w for w in col_widths]
        lines.append("| " + " | ".join(separator_cells) + " |")
        
        # Build data rows
        for row in table.rows:
            cells = [
                str(cell).ljust(col_widths[i])
                for i, cell in enumerate(row.cells[:num_cols])
            ]
            lines.append("| " + " | ".join(cells) + " |")
        
        return "\n".join(lines)

    def _render_bullets(self, items: list) -> str:
        """Render a bullet list."""
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)

    def _render_code(self, language: str, code: str) -> str:
        """Render a code block with language specification."""
        if not code:
            return ""
        return f"```{language}\n{code}\n```"
