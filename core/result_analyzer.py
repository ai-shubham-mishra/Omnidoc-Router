"""
Reads ENTIRE workflow response and generates curated insights.
Uses Azure OpenAI json_schema mode for structured responses.
"""
import json
import logging
from typing import Any, Dict, Optional, Literal

from core.azure_openai_client import AzureOpenAIClient
from models.response_schemas import StructuredResponse

logger = logging.getLogger(__name__)

# Sensitive fields to strip before LLM analysis (security + privacy)
SENSITIVE_FIELDS = {
    "userId", "user_id", "orgId", "org_id", 
    "workflowId", "workflow_id", "fileId", "file_id",
    "pending_workflow_run_id", "sessionId", "session_id",
    "runId", "run_id", "_id", "mongoPoId", "mongo_po_id",
    "userEmail", "user_email", "createdBy", "created_by",
}


class ResultAnalyzer:
    """Analyzes workflow responses and returns structured insights."""

    def __init__(self, llm_client: Optional[AzureOpenAIClient] = None):
        self.llm = llm_client or AzureOpenAIClient()
    
    def _sanitize_sensitive_data(self, data: Any) -> Any:
        """
        Recursively remove sensitive fields (IDs, emails) from data.
        Prevents leaking backend identifiers to LLM responses.
        """
        if not isinstance(data, dict):
            return data
        
        clean_data = {}
        for key, value in data.items():
            # Skip sensitive keys entirely
            if key in SENSITIVE_FIELDS:
                logger.debug(f"Sanitized sensitive field: {key}")
                continue
            
            # Recurse into nested dicts
            if isinstance(value, dict):
                clean_data[key] = self._sanitize_sensitive_data(value)
            
            # Recurse into lists
            elif isinstance(value, list):
                clean_data[key] = [
                    self._sanitize_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            
            else:
                clean_data[key] = value
        
        return clean_data
    
    def _suggest_format(self, data: Dict[str, Any]) -> Literal["paragraph", "bullets", "table"]:
        """
        Analyze data structure and suggest optimal presentation format.
        Prevents LLM from defaulting to tables for everything.
        
        Returns:
            "paragraph" - Simple results (1-3 values)
            "bullets" - List of items or findings (4-8 items)
            "table" - Multi-dimensional comparison (3+ items × 2+ attributes)
        """
        if not data:
            return "paragraph"
        
        # Count top-level metrics/fields
        num_fields = len(data)
        
        # Check if values are lists (multiple items per field)
        has_lists = any(isinstance(v, list) for v in data.values())
        
        # Check if values are nested dicts (multi-dimensional)
        has_nested_dicts = any(isinstance(v, dict) for v in data.values())
        
        # Decision logic
        if num_fields <= 3 and not has_nested_dicts:
            # Simple key-value pairs → paragraph
            # "PO Number: X, Total: Y, Status: Z"
            return "paragraph"
        
        elif has_lists and not has_nested_dicts and num_fields <= 6:
            # List of items per field → bullets
            # "Recommendations: [A, B, C]" → bullet list
            return "bullets"
        
        elif has_nested_dicts or num_fields > 8:
            # Complex multi-dimensional data → table
            # {channel_a: {roas: X, spend: Y}, channel_b: {...}} → table
            return "table"
        
        else:
            # Mid-range complexity → bullets (safe default)
            return "bullets"
    
    def analyze_workflow_result(
        self,
        result: Any,
        workflow_name: str,
        result_type: str = "final"  # "final" or "hitl"
    ) -> StructuredResponse:
        """
        Analyze complete workflow result and generate structured response.
        
        Args:
            result: Full workflow response dict
            workflow_name: Name of workflow
            result_type: "final" for completion, "hitl" for confirmation
            
        Returns:
            StructuredResponse object for markdown building
        """
        if not isinstance(result, dict):
            # Fallback for non-dict results
            return self._format_simple_result(result, workflow_name, result_type)
        
        # Extract all components of the response
        status = result.get("status", "unknown")
        message = result.get("message", "")
        data = result.get("data", {}) or result.get("analytics_data", {}) or result.get("body", {})
        file_outputs = result.get("file_outputs", [])
        
        # SECURITY: Sanitize sensitive fields before LLM analysis
        data = self._sanitize_sensitive_data(data)
        
        # QUALITY: Suggest optimal format based on data structure
        suggested_format = self._suggest_format(data) if data else "paragraph"
        
        # Build context for LLM
        if result_type == "hitl":
            # For HITL: Show only the extracted data, not status/message wrappers
            context = self._build_hitl_context(data, file_outputs)
        else:
            # For final results: Show everything including status/message
            context = self._build_analysis_context(
                status=status,
                message=message,
                data=data,
                file_outputs=file_outputs,
                workflow_name=workflow_name
            )
        
        # Generate analysis based on type
        if result_type == "hitl":
            return self._generate_hitl_summary(context, workflow_name, suggested_format)
        else:
            return self._generate_final_summary(context, workflow_name, suggested_format)
    
    def _build_hitl_context(
        self,
        data: Dict[str, Any],
        file_outputs: list
    ) -> str:
        """
        Build context for HITL confirmation - ONLY the extracted data.
        Excludes status/message wrappers to avoid robotic echoing.
        """
        context_parts = []
        
        # Only show the actual extracted data
        if data:
            data_str = json.dumps(data, indent=2, default=str)
            # Truncate if too long
            if len(data_str) > 4000:
                data_str = data_str[:4000] + "\n... (truncated)"
            context_parts.append(f"**Extracted Data:**\n```json\n{data_str}\n```")
        
        # NOTE: File outputs intentionally excluded from HITL context
        # Files are handled separately in the JSON response structure
        
        return "\n\n".join(context_parts) if context_parts else "No data extracted."
    
    def _build_analysis_context(
        self,
        status: str,
        message: str,
        data: Dict[str, Any],
        file_outputs: list,
        workflow_name: str
    ) -> str:
        """Build comprehensive context string for LLM analysis."""
        context_parts = []
        
        # Status
        context_parts.append(f"**Status:** {status}")
        
        # Message (if meaningful)
        if message and message != "success":
            context_parts.append(f"**Message:** {message}")
        
        # Data analysis
        if data:
            data_str = json.dumps(data, indent=2, default=str)
            # Truncate if too long
            if len(data_str) > 4000:
                data_str = data_str[:4000] + "\n... (truncated)"
            context_parts.append(f"**Data:**\n```json\n{data_str}\n```")
        
        # NOTE: File outputs are intentionally excluded from LLM context
        # Files are handled separately in the JSON response structure
        
        return "\n\n".join(context_parts)
    
    def _generate_final_summary(
        self,
        context: str,
        workflow_name: str,
        suggested_format: str = "bullets"
    ) -> StructuredResponse:
        """Generate final result summary — structured insights only, no follow-ups."""
        
        # Format-specific guidance
        format_guidance = {
            "paragraph": """
SUGGESTED FORMAT: Paragraph
Use a paragraph format since the data is simple (1-3 key values).
Example: "The extraction completed successfully. PO Number: **PO-2024-0042**, Supplier: **Acme Corp**, Total: **€12,500**."
Only use table if comparing 3+ items across 2+ dimensions.""",
            
            "bullets": """
SUGGESTED FORMAT: Bullet List
Use bullets since you have a list of distinct findings/items.
Example:
- Channel A outperformed targets by 15%
- Channel B needs optimization
- Total campaign ROI: 124%
Only use table if comparing multiple items across 2+ attributes.""",
            
            "table": """
SUGGESTED FORMAT: Table
Use a table to compare multiple items across dimensions.
Example: Channel performance (5 channels × metrics like ROAS, Spend, Conversions).
Avoid tables for simple 1-2 row data — use paragraph instead."""
        }
        
        prompt = f"""Analyze the complete result of the "{workflow_name}" workflow.

FULL WORKFLOW RESPONSE:
{context}

{format_guidance.get(suggested_format, format_guidance["bullets"])}

INSTRUCTIONS:
1. Read the data and metrics from the response.
2. Extract key outcomes and interpret what they MEAN, not just echo values.
   - Example: Instead of "Total: 50000", say "Generated $50,000 in revenue"
   - **CRITICAL FOR ROAS/Marginal ROAS**: Present as decimal values rounded to 2 decimal places, NOT percentages
     ✅ Correct: "Channel A ROAS: **1.25**" or "Channel A returned **1.25x** on ad spend"
     ❌ Wrong: "Channel A ROAS: 125%" or "Channel A: 124.64%"
   - **CRITICAL FOR WATERFALL/CONTRIBUTION PERCENTAGES**: The percentage field is ALREADY a percentage (0-100+ scale).
     Do NOT multiply by 100. Round to 2 decimals WITHOUT the % symbol (the column header already indicates %).
     ✅ Correct: If percentage value is 44.69955488804548, display as **44.70** (NOT 44.70% or 4,469.96%)
     ✅ Correct: If percentage value is 4.640157566353377, display as **4.64** (NOT 4.64% or 464.02%)
     ❌ Wrong: Adding % symbol, or multiplying by 100
3. Choose the most natural format based on data complexity:
   - **Paragraph**: Simple results with 1-3 values
   - **Bullets**: List of 4-8 distinct findings or items
   - **Table**: ONLY for comparing 3+ items across 2+ dimensions
4. Use headings (level 2 or 3) to organize major topics if needed.
5. Bold important values (metrics, amounts, counts, statuses).

ANTI-PATTERNS (avoid these):
❌ Table with only 1-2 rows → Use paragraph instead
❌ Table with only 1 column → Use bullets instead
❌ Bullets for 2-3 simple values → Use paragraph instead

STRICTLY FORBIDDEN — your response MUST NOT contain:
- Any mention of generated files, visualization files, or file outputs
- Sections or headings about "Files Generated", "Visualization Files", "Downloads", etc.
- File counts, file names, or file references of any kind
- Follow-up questions ("What would you like to do next?", "Would you like to...", etc.)
- Suggestions or calls-to-action ("Let me know if...", "Feel free to...")
- Advice, recommendations, or domain-specific analysis
- Mentions of specific file paths or file systems
- Raw JSON dumps, code fences with JSON data, or system commentary
- Repetition of the status/message field verbatim
- Vague language like "some data" or "various metrics"

Focus ONLY on the data insights and metrics. Files are handled separately by the system.

Return a structured response with clear sections that make the results actionable and easy to understand."""

        try:
            return self.llm.call_with_structured_output(
                prompt=prompt,
                response_format=StructuredResponse,
                temperature=0.3,
                max_tokens=1536
            )
        except Exception as e:
            logger.error(f"Result analysis failed: {e}")
            return self._format_simple_result(context, workflow_name, "final")
    
    def _generate_hitl_summary(
        self,
        context: str,
        workflow_name: str,
        suggested_format: str = "bullets"
    ) -> StructuredResponse:
        """Generate HITL review prompt — structured data for confirmation."""
        
        # Format-specific guidance for HITL
        format_guidance = {
            "paragraph": """
SUGGESTED FORMAT: Paragraph
The extracted data is simple (1-3 fields). Present as a natural paragraph.
Example: "I've extracted the PO details: Number **PO-2024-0042**, Supplier **Acme Corp**, Total **€12,500**."
""",
            "bullets": """
SUGGESTED FORMAT: Bullet List
Present the extracted fields as a clear bullet list.
Example:
"Here's what I extracted:
- PO Number: **PO-2024-0042**
- Supplier: **Acme Corp**
- Total Amount: **€12,500**"
""",
            "table": """
SUGGESTED FORMAT: Table
Use a table to present the extracted key-value pairs with multiple fields.
"""
        }
        
        prompt = f"""The "{workflow_name}" workflow extracted data that needs user confirmation.

EXTRACTED DATA:
{context}

{format_guidance.get(suggested_format, format_guidance["bullets"])}

INSTRUCTIONS:
1. Read the extracted data and present it naturally in plain English.
2. DO NOT echo "Status is..." or "Message is..." — focus only on the actual data fields.
3. Use natural language:
   - ✅ "I've extracted the following PO details:"
   - ✅ "Here's what was found in the document:"
   - ❌ "Extracted: **Status** is **waiting**, **Message** is..."
4. Choose the most natural format:
   - **Paragraph**: For simple data (1-3 fields)
   - **Table**: For structured key-value data with 4+ fields
   - **Bullets**: For a list of extracted items
5. Bold important values (numbers, names, amounts, dates).
6. End with a simple confirmation request: "Please review and confirm to proceed."

STRICTLY FORBIDDEN:
- Echoing system fields like "status", "message", or internal metadata
- Robotic phrasing like "Extracted: **Status** is..."
- Any mention of files, file outputs, or visualizations
- Follow-up questions beyond the confirmation request
- Mentions of file paths or JSON structure
- Vague language ("I extracted some data...")

Focus ONLY on the extracted data values that need user review. Return a structured response."""

        try:
            return self.llm.call_with_structured_output(
                prompt=prompt,
                response_format=StructuredResponse,
                temperature=0.3,
                max_tokens=1024
            )
        except Exception as e:
            logger.error(f"HITL analysis failed: {e}")
            # Fallback: simple structured response
            from models.response_schemas import ContentSection
            return StructuredResponse(sections=[
                ContentSection(
                    type="paragraph",
                    text="Extracted data is ready for review. Please confirm to proceed."
                )
            ])
    
    def _format_simple_result(
        self, 
        result: Any, 
        workflow_name: str, 
        result_type: str
    ) -> StructuredResponse:
        """Fallback formatter for non-dict or error results."""
        from models.response_schemas import ContentSection
        
        if isinstance(result, dict):
            if result.get("status") == "success":
                msg = f"**{workflow_name}** completed successfully."
                
                return StructuredResponse(sections=[
                    ContentSection(type="paragraph", text=msg)
                ])
            
            elif result.get("status") == "error":
                error = result.get("error", {})
                detail = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                return StructuredResponse(sections=[
                    ContentSection(type="paragraph", text=f"**Error:** {detail}")
                ])
        
        if result_type == "hitl":
            msg = "Extracted data is ready for review. Please confirm to proceed."
        else:
            msg = f"**{workflow_name}** completed."
        
        return StructuredResponse(sections=[
            ContentSection(type="paragraph", text=msg)
        ])
