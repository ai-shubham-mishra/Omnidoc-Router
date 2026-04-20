"""
File Intelligence Service for the LLM Router.
Handles file classification, smart matching, and context-aware file management.

Phase 1: Stop blind auto-fill (file lifecycle)
Phase 2: File classification via Gemini
Phase 3: Smart file-to-input matching
Phase 4: File context for RAG-style queries
"""
import os
import json
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv
from components.KeyVaultClient import get_secret

load_dotenv()
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY", default=os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key=GOOGLE_API_KEY)

# File type to MIME type mapping for fileType validation
FILE_TYPE_MIME_MAP = {
    "json": ["application/json"],
    "pdf": ["application/pdf"],
    "csv": ["text/csv", "application/csv"],
    "document": [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",        # .xlsx
        "application/vnd.openxmlformats-officedocument.presentationml.presentation", # .pptx
        "application/msword",  # .doc
        "application/vnd.ms-excel",  # .xls
        "application/vnd.ms-powerpoint",  # .ppt
    ],
    "image": [
        "image/jpeg", "image/jpg", "image/png", 
        "image/gif", "image/webp", "image/bmp"
    ],
    "email": ["message/rfc822", "application/vnd.ms-outlook"],
    "text": ["text/plain"],
    "xml": ["application/xml", "text/xml"],
    "any": ["*"]  # Wildcard - accepts all file types
}


class FileIntelligence:
    """
    Intelligent file management for multi-workflow sessions.
    Classifies files, matches them to workflow inputs, and tracks ownership.
    """

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )

    # ==================== Phase 2: File Classification ====================

    def classify_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a file by its name, mime type, and extension.
        Uses Gemini to infer the document type and generate a short summary.
        Falls back to basic classification if Gemini is unavailable (rate limits).

        Args:
            file_info: Dict with original_name, mime_type, size_bytes

        Returns:
            Classification dict with document_type, summary, keywords, confidence
        """
        original_name = file_info.get("original_name", "unknown")
        mime_type = file_info.get("mime_type", "application/octet-stream")
        size_bytes = file_info.get("size_bytes", 0)

        # Try Gemini classification with retry
        for attempt in range(2):
            try:
                prompt = f"""Classify this uploaded document based on its filename and type.

Filename: "{original_name}"
MIME type: {mime_type}
Size: {size_bytes} bytes

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.
Return format:
{{
  "document_type": "<type such as: purchase_order, pricesheet, json_data, invoice, business_card, resume, contract, receipt, report, spreadsheet, image, presentation, letter, form, id_document, certificate, unknown>",
  "summary": "<1-sentence description of what this file likely contains>",
  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>"],
  "confidence": <0.0-1.0>
}}"""

                response = self.model.generate_content(prompt)
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

                classification = json.loads(text)
                logger.info(
                    f"Classified '{original_name}' as '{classification.get('document_type')}' "
                    f"(confidence: {classification.get('confidence', 0):.0%})"
                )
                return classification

            except Exception as e:
                if "429" in str(e) or "Resource exhausted" in str(e):
                    logger.warning(f"Gemini rate limit hit (attempt {attempt + 1}/2), falling back to basic classification")
                    break  # Don't retry on rate limit
                elif attempt == 0:
                    time.sleep(1)  # Brief retry for transient errors
                    continue
                else:
                    logger.warning(f"File classification failed for '{original_name}': {e}")
                    break

        # Fallback: Basic classification from filename patterns
        return self._fallback_classify(original_name, mime_type)

    def _fallback_classify(self, filename: str, mime_type: str) -> Dict[str, Any]:
        """Basic filename-based classification when Gemini is unavailable."""
        filename_lower = filename.lower()
        
        # Pattern matching for common document types
        if any(kw in filename_lower for kw in ["po", "purchase", "order", "procurement"]):
            return {
                "document_type": "purchase_order",
                "summary": f"Purchase order document: {filename}",
                "keywords": ["purchase_order", "po", "procurement"],
                "confidence": 0.7,
            }
        elif any(kw in filename_lower for kw in ["price", "pricesheet", "pricing", "pricelist"]):
            return {
                "document_type": "pricesheet",
                "summary": f"Pricesheet or pricing document: {filename}",
                "keywords": ["pricesheet", "pricing", "price", "json"],
                "confidence": 0.8,
            }
        elif any(kw in filename_lower for kw in ["invoice", "bill", "receipt"]):
            return {
                "document_type": "invoice",
                "summary": f"Invoice or billing document: {filename}",
                "keywords": ["invoice", "billing"],
                "confidence": 0.7,
            }
        elif any(kw in filename_lower for kw in ["contract", "agreement"]):
            return {
                "document_type": "contract",
                "summary": f"Contract or agreement: {filename}",
                "keywords": ["contract", "agreement"],
                "confidence": 0.7,
            }
        elif mime_type.startswith("image/"):
            return {
                "document_type": "image",
                "summary": f"Image file: {filename}",
                "keywords": ["image"],
                "confidence": 0.8,
            }
        elif mime_type == "application/json" or filename_lower.endswith(".json"):
            return {
                "document_type": "json_data",
                "summary": f"JSON data file: {filename}",
                "keywords": ["json", "data", "pricesheet" if "price" in filename_lower else "config"],
                "confidence": 0.8,
            }
        elif "spreadsheet" in mime_type or filename_lower.endswith((".xlsx", ".xls", ".csv")):
            return {
                "document_type": "spreadsheet",
                "summary": f"Spreadsheet document: {filename}",
                "keywords": ["spreadsheet", "data"],
                "confidence": 0.8,
            }
        else:
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            return {
                "document_type": "unknown",
                "summary": f"Uploaded file: {filename}",
                "keywords": [ext] if ext else [],
                "confidence": 0.5,
            }

    # ==================== Phase 3: Smart File Matching ====================

    def _validate_mime_type(
        self, 
        file_info: Dict[str, Any], 
        expected_file_type: str
    ) -> bool:
        """
        Validate if file's MIME type matches expected fileType.
        
        Args:
            file_info: File metadata with mime_type
            expected_file_type: Expected file type from workflow schema (e.g., "json", "pdf")
            
        Returns:
            True if valid match, False if mismatch
        """
        actual_mime = file_info.get("mime_type", "").lower()
        expected_mimes = FILE_TYPE_MIME_MAP.get(expected_file_type, [])
        
        if not expected_mimes:
            # Unrecognized file type - log warning and allow
            logger.warning(f"Unrecognized fileType '{expected_file_type}', treating as 'any'")
            return True
        
        # Wildcard check
        if "*" in expected_mimes:
            return True
        
        # Exact match check
        if actual_mime in expected_mimes:
            return True
        
        # Partial match for generic MIME types (e.g., "image/*" matches "image/png")
        for expected_mime in expected_mimes:
            if expected_mime.endswith("/*"):
                mime_category = expected_mime.replace("/*", "")
                if actual_mime.startswith(mime_category + "/"):
                    return True
        
        return False

    def match_file_to_input(
        self,
        file_info: Dict[str, Any],
        input_spec: Dict[str, Any],
        workflow_name: str,
    ) -> float:
        """
        Score how well a file matches a specific workflow input.
        Now includes MIME type validation as a hard requirement.

        Args:
            file_info: File metadata including classification and mime_type
            input_spec: Workflow input spec (field, type, label, file_type)
            workflow_name: Name of the target workflow

        Returns:
            Score from -100.0 to 10.0 (higher = better match, -100 = hard rejected)
        """
        # ✨ TIER 0: MIME Type Hard Validation (NEW)
        expected_file_type = input_spec.get("file_type")
        if expected_file_type and expected_file_type != "any":
            if not self._validate_mime_type(file_info, expected_file_type):
                # HARD BLOCK - MIME type mismatch
                logger.info(
                    f"❌ MIME type mismatch: '{file_info.get('original_name')}' "
                    f"(type: {file_info.get('mime_type')}) rejected for input '{input_spec.get('label')}' "
                    f"(expects: {expected_file_type})"
                )
                return -100.0
        
        # Continue with existing scoring logic
        score = 0.0
        classification = file_info.get("classification", {})
        doc_type = classification.get("document_type", "unknown")
        doc_keywords = classification.get("keywords", [])
        input_label = input_spec.get("label", "").lower()
        input_field = input_spec.get("field", "").lower()
        wf_name_lower = workflow_name.lower()

        # 1. Document type matches workflow/input name
        type_matches = {
            "purchase_order": ["po", "purchase", "order", "procurement", "document"],
            "pricesheet": ["price", "pricesheet", "pricing", "pricelist", "json"],
            "json_data": ["json", "data", "price", "pricesheet", "config"],
            "invoice": ["invoice", "billing", "payment"],
            "business_card": ["business card", "contact", "lead", "hubspot", "crm"],
            "resume": ["resume", "cv", "candidate", "recruitment", "hiring"],
            "contract": ["contract", "agreement", "legal"],
            "receipt": ["receipt", "expense", "reimbursement"],
            "id_document": ["identity", "id", "verification", "kyc"],
            "certificate": ["certificate", "certification", "credential"],
        }

        matched_type = False
        for doc_cat, match_words in type_matches.items():
            if doc_type == doc_cat:
                matched_type = True
                if any(w in input_label or w in wf_name_lower for w in match_words):
                    score += 5.0
                else:
                    # Doc type identified but doesn't match this workflow — strong negative
                    score -= 3.0
                break

        # Unknown doc type gets no type bonus — must rely on other signals
        if doc_type == "unknown" and not file_info.get("uploaded_during_workflow"):
            score -= 1.0

        # 2. Keyword overlap between file and input/workflow
        combined_target = f"{input_label} {wf_name_lower} {input_field}"
        for kw in doc_keywords:
            if kw.lower() in combined_target:
                score += 1.5

        # 3. File already used by a completed workflow (penalize reuse)
        used_workflows = file_info.get("used_by_workflows", [])
        if used_workflows:
            score -= 3.0

        # 4. File status
        if file_info.get("status") == "used":
            score -= 2.0

        # 5. Bonus for files uploaded AFTER current workflow was identified
        if file_info.get("uploaded_during_workflow"):
            if file_info["uploaded_during_workflow"] == workflow_name:
                score += 3.0

        return max(0.0, score)

    def find_best_file_for_input(
        self,
        session_files: List[Dict[str, Any]],
        input_spec: Dict[str, Any],
        workflow_name: str,
        auto_fill_threshold: float = 4.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best matching file from session for a workflow input.
        Only considers available files (not used/inaccessible).

        Args:
            session_files: All files in the session
            input_spec: The workflow input that needs a file
            workflow_name: Name of the current workflow
            auto_fill_threshold: Minimum score to auto-fill (default 4.0)

        Returns:
            Best matching file info, or None if no confident match
        """
        if not session_files:
            return None

        candidates = []
        for f in session_files:
            if not f.get("accessible", True):
                continue
            if f.get("status", "available") != "available":
                continue
            if not f.get("stored_path"):
                continue

            score = self.match_file_to_input(f, input_spec, workflow_name)
            candidates.append((f, score))

        if not candidates:
            return None

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_file, best_score = candidates[0]

        logger.info(
            f"📎 Best file match for '{input_spec.get('label', input_spec.get('field'))}': "
            f"'{best_file.get('original_name')}' (score: {best_score:.1f}, threshold: {auto_fill_threshold})"
        )

        if best_score >= auto_fill_threshold:
            return best_file

        return None

    # ==================== Phase 4: Session Context Builder ====================

    def build_session_context(
        self,
        session: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a rich session context summary for Gemini.
        Used for RAG-style conversational awareness and question answering.

        Args:
            session: Full session dict from MongoDB/Redis

        Returns:
            Context dict with workflow history (including results), files, and current state
        """
        workflow_history = session.get("workflow_history", [])
        uploaded_files = session.get("uploaded_files", [])
        current_wf = session.get("current_workflow", {})
        conversation = session.get("conversation_history", [])

        # Summarize completed workflows WITH result data
        completed = []
        for wf in workflow_history[-5:]:  # Last 5 workflows
            wf_entry = {
                "name": wf.get("workflow_name"),
                "status": wf.get("status"),
                "completed_at": wf.get("completed_at"),
            }

            # Extract key result data for question answering
            result = wf.get("result")
            if result and isinstance(result, dict):
                result_summary = wf.get("result_summary")
                if result_summary:
                    wf_entry["result_summary"] = result_summary
                else:
                    # Build summary from raw result (truncated for context size)
                    flat = self._flatten_result(result)
                    if flat:
                        wf_entry["result_data"] = flat

            completed.append(wf_entry)

        # Summarize files with classification
        files_summary = []
        for f in uploaded_files:
            classification = f.get("classification", {})
            files_summary.append({
                "file_id": f.get("file_id"),
                "name": f.get("original_name"),
                "type": classification.get("document_type", "unknown"),
                "summary": classification.get("summary", ""),
                "status": f.get("status", "available"),
                "used_by": f.get("used_by_workflows", []),
            })

        # Current workflow state with collected input details
        current = None
        if current_wf.get("workflow_id"):
            required = current_wf.get("required_inputs", [])
            collected_inputs_dict = current_wf.get("collected_inputs", {})

            missing_inputs = [
                inp.get("label", inp.get("field"))
                for inp in required
                if not inp.get("collected")
            ]

            # Include collected inputs with their actual values/filenames
            collected_details = []
            for inp in required:
                if inp.get("collected"):
                    label = inp.get("label", inp.get("field"))
                    value = collected_inputs_dict.get(inp.get("field"))
                    detail = {"label": label}
                    if isinstance(value, list):
                        # File paths — extract filenames
                        detail["files"] = [os.path.basename(v) for v in value if isinstance(v, str)]
                    elif isinstance(value, str) and len(value) < 200:
                        detail["value"] = value
                    collected_details.append(detail)

            current = {
                "name": current_wf.get("workflow_name"),
                "status": current_wf.get("status"),
                "missing_inputs": missing_inputs,
                "collected_inputs": collected_details,
            }

        # Recent conversation (last 15 turns for better context)
        recent_messages = []
        for msg in conversation[-15:]:
            recent_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content", "")[:300],  # Slightly longer truncation
            })

        return {
            "completed_workflows": completed,
            "current_workflow": current,
            "session_files": files_summary,
            "recent_messages": recent_messages,
            "total_files": len(uploaded_files),
            "total_workflows_completed": len(workflow_history),
        }

    def _flatten_result(self, result: Dict[str, Any], max_keys: int = 20) -> Dict[str, Any]:
        """
        Extract key-value pairs from a nested workflow result for context.
        Keeps it compact enough for Gemini context window.
        """
        flat = {}
        count = 0

        def _extract(obj, prefix=""):
            nonlocal count
            if count >= max_keys:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if count >= max_keys:
                        return
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, (str, int, float, bool)):
                        flat[key] = v
                        count += 1
                    elif isinstance(v, dict):
                        _extract(v, key)
                    elif isinstance(v, list) and len(v) <= 5:
                        flat[key] = v
                        count += 1

        _extract(result)
        return flat

    def answer_session_question(
        self,
        question: str,
        session_context: Dict[str, Any],
    ) -> str:
        """
        Answer a user's question about the session using context.
        Now enhanced with Pinecone semantic search for long-term memory.
        Handles questions like:
        - "Which document are you referring to?"
        - "What files have I uploaded?"
        - "What happened with the PO?"
        - "What did I upload last week?"

        Args:
            question: User's question
            session_context: Context from build_session_context() + Pinecone context

        Returns:
            Natural language answer
        """
        # Extract Pinecone context if available
        pinecone_context = session_context.pop("pinecone_context", None)
        
        context_text = json.dumps(session_context, indent=2, default=str)
        
        # Add Pinecone context separately for clarity
        pinecone_hint = ""
        if pinecone_context:
            relevant_messages = pinecone_context.get("relevant_messages", [])
            relevant_files = pinecone_context.get("relevant_files", [])
            
            if relevant_messages or relevant_files:
                pinecone_hint = "\n\nRelevant past context (from semantic search):\n"
                if relevant_messages:
                    pinecone_hint += "Messages:\n"
                    for msg in relevant_messages[:3]:
                        pinecone_hint += f"  - [{msg['role']}]: {msg['content'][:100]}...\n"
                if relevant_files:
                    pinecone_hint += "Files:\n"
                    for f in relevant_files[:3]:
                        pinecone_hint += f"  - {f['filename']}: {f['content_preview'][:80]}...\n"
        
        prompt = f"""You are a helpful workflow assistant. Answer the user's question based on the current session context and past relevant context.

User question: "{question}"

Current session context:
{context_text}{pinecone_hint}

Rules:
- Be concise (2-4 sentences max)
- Reference specific files by their original name when relevant
- If asking about a file, explain its document type and which workflow it was used for (or if it's still available)
- If asking about current workflow status, explain what inputs are collected (with filenames if applicable) and what's still missing
- If asking about past workflow results (e.g., "what was the PO number?", "what was extracted?"), look in the completed_workflows result_data or result_summary fields and quote specific values
- Use the semantic search results to answer questions about past uploads or conversations beyond current session
- Distinguish between files used by completed workflows (status: "used") and files still available for new workflows (status: "available")
- If the context contains result data from completed workflows, use it to answer factual questions about those results
- If the question can't be answered from context, say so honestly and suggest the user check the workflow output directly
- Do NOT make up information not in the context
- Do NOT suggest re-running a workflow to answer a question about past results"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Session question answering failed: {e}")
            return "I'm sorry, I couldn't process your question. Could you try rephrasing?"

    # ==================== Phase 1: File Lifecycle Helpers ====================

    def mark_files_used(
        self,
        session_files: List[Dict[str, Any]],
        used_file_paths: List[str],
        workflow_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Mark specific files as 'used' by a workflow.
        Returns updated file list (doesn't persist — caller must save).

        Args:
            session_files: All session files
            used_file_paths: File paths that were used
            workflow_name: Name of the workflow that used them

        Returns:
            Updated file list with usage tracking
        """
        for f in session_files:
            if f.get("stored_path") in used_file_paths:
                f["status"] = "used"
                if "used_by_workflows" not in f:
                    f["used_by_workflows"] = []
                if workflow_name not in f["used_by_workflows"]:
                    f["used_by_workflows"].append(workflow_name)
                logger.info(f"📎 Marked '{f.get('original_name')}' as used by '{workflow_name}'")
        return session_files

    def get_available_files(
        self,
        session_files: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Get files that haven't been used by any workflow yet."""
        return [
            f for f in session_files
            if f.get("status", "available") == "available"
            and f.get("accessible", True)
            and f.get("stored_path")
        ]
