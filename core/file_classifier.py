"""
File Classifier - Intelligent file-to-workflow matching using hybrid approach.

3-Phase Classification Strategy:
1. Fast Signature Matching (filename, MIME, patterns) - < 100ms
2. OCR + Keyword Analysis (text extraction) - 2-5s
3. LLM Disambiguation (context-aware) - 3-10s

Uses file_signature from workflow schema for accurate matching.
"""
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from components.FileMiddleware import file_middleware
from components.OCRExtraction import extract_text_from_file
from core.azure_openai_client import AzureOpenAIClient

logger = logging.getLogger(__name__)


class FileClassifier:
    """Intelligent file classification for idle workflows."""
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.85
    MEDIUM_CONFIDENCE = 0.60
    
    # Phase 1 scoring weights
    FILENAME_MATCH_SCORE = 40
    MIME_TYPE_SCORE = 20
    RECENCY_BOOST = 10
    
    # Phase 2 scoring weights
    KEYWORD_SCORE = 15
    OCR_PATTERN_SCORE = 25
    
    def __init__(self):
        self.llm = AzureOpenAIClient()
    
    async def classify_file(
        self,
        file_id: str,
        idle_workflows: List[Dict[str, Any]],
        conversation_context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main classification entry point.
        
        Args:
            file_id: File to classify
            idle_workflows: List of workflows awaiting files
            conversation_context: Recent user messages for context
        
        Returns:
            {
                "matched": bool,
                "workflow_instance_id": str | None,
                "input_field": str | None,
                "confidence": float,
                "reasoning": str,
                "candidates": List[Dict] (if ambiguous)
            }
        """
        if not idle_workflows:
            return {
                "matched": False,
                "workflow_instance_id": None,
                "input_field": None,
                "confidence": 0.0,
                "reasoning": "No idle workflows available",
                "candidates": []
            }
        
        # Extract file features
        features = await self._extract_file_features(file_id)
        if not features:
            return {
                "matched": False,
                "workflow_instance_id": None,
                "input_field": None,
                "confidence": 0.0,
                "reasoning": "Could not extract file features",
                "candidates": []
            }
        
        # Phase 1: Fast Signature Matching
        phase1_results = await self._phase1_signature_matching(features, idle_workflows)
        
        # Check for high confidence match
        if phase1_results["best_score"] >= 70:
            best = phase1_results["best_match"]
            return {
                "matched": True,
                "workflow_instance_id": best["instance_id"],
                "input_field": best["field"],
                "confidence": min(phase1_results["best_score"] / 100.0, 1.0),
                "reasoning": f"High confidence filename/MIME match: {best['workflow_name']}",
                "candidates": []
            }
        
        # Phase 2: OCR + Keyword Analysis
        if phase1_results["best_score"] >= 30:  # Has potential candidates
            phase2_results = await self._phase2_ocr_analysis(
                features,
                phase1_results["candidates"],
                file_id
            )
            
            # Check for strong OCR match
            if phase2_results["best_score"] >= 85:
                best = phase2_results["best_match"]
                return {
                    "matched": True,
                    "workflow_instance_id": best["instance_id"],
                    "input_field": best["field"],
                    "confidence": min(phase2_results["best_score"] / 100.0, 1.0),
                    "reasoning": f"Strong OCR content match: {best['workflow_name']}",
                    "candidates": []
                }
            
            # Check for ambiguous (multiple candidates 70-85)
            strong_candidates = [c for c in phase2_results["candidates"] if c["score"] >= 70]
            if len(strong_candidates) > 1:
                # Phase 3: LLM Disambiguation
                phase3_result = await self._phase3_llm_disambiguation(
                    features,
                    strong_candidates,
                    conversation_context or []
                )
                return phase3_result
            
            # Single strong candidate
            if len(strong_candidates) == 1:
                best = strong_candidates[0]
                return {
                    "matched": True,
                    "workflow_instance_id": best["instance_id"],
                    "input_field": best["field"],
                    "confidence": min(best["score"] / 100.0, 1.0),
                    "reasoning": f"OCR analysis match: {best['workflow_name']}",
                    "candidates": []
                }
        
        # Low confidence - no match
        return {
            "matched": False,
            "workflow_instance_id": None,
            "input_field": None,
            "confidence": phase1_results["best_score"] / 100.0 if phase1_results["best_score"] > 0 else 0.0,
            "reasoning": "No confident match found. Please specify which workflow this file belongs to.",
            "candidates": phase1_results["candidates"][:3]  # Top 3 for user choice
        }
    
    async def _extract_file_features(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Extract features from file metadata."""
        try:
            metadata = file_middleware.get_files_metadata([file_id])
            if not metadata:
                logger.error(f"File metadata not found for {file_id}")
                return None
            
            file_meta = metadata[0]
            
            return {
                "file_id": file_id,
                "filename": file_meta.get("original_name", ""),
                "mime_type": file_meta.get("mime_type", ""),
                "size_bytes": file_meta.get("size_bytes", 0),
                "uploaded_at": file_meta.get("uploaded_at", ""),
            }
        except Exception as e:
            logger.error(f"Error extracting file features: {e}")
            return None
    
    async def _phase1_signature_matching(
        self,
        features: Dict[str, Any],
        idle_workflows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Phase 1: Fast signature matching using filename and MIME type.
        
        Returns:
            {
                "best_score": float,
                "best_match": Dict | None,
                "candidates": List[Dict]
            }
        """
        filename = features["filename"].lower()
        mime_type = features["mime_type"]
        
        candidates = []
        
        for workflow in idle_workflows:
            for missing_input in workflow.get("missing_inputs", []):
                score = 0.0
                field = missing_input["field"]
                signature = missing_input.get("file_signature", {})
                
                if not signature:
                    # No signature - generic low score
                    score = 5.0
                else:
                    # Filename pattern matching
                    filename_patterns = signature.get("filename_patterns", [])
                    for pattern in filename_patterns:
                        pattern_regex = pattern.replace("*", ".*").lower()
                        if re.search(pattern_regex, filename):
                            score += self.FILENAME_MATCH_SCORE
                            break
                    
                    # MIME type validation
                    allowed_mimes = signature.get("mime_types", [])
                    if mime_type in allowed_mimes:
                        score += self.MIME_TYPE_SCORE
                    elif allowed_mimes and mime_type not in allowed_mimes:
                        # Strict MIME enforcement - disqualify
                        score = 0.0
                        continue
                    
                    # Exclude patterns
                    exclude_patterns = signature.get("exclude_patterns", [])
                    for exclude in exclude_patterns:
                        if exclude.lower() in filename:
                            score = 0.0
                            break
                    
                    if score == 0.0:
                        continue
                    
                    # Recency boost
                    last_active = workflow.get("last_active", "")
                    if last_active:
                        try:
                            last_active_dt = datetime.fromisoformat(last_active)
                            now = datetime.utcnow()
                            minutes_ago = (now - last_active_dt).total_seconds() / 60
                            if minutes_ago < 5:
                                score += self.RECENCY_BOOST
                        except:
                            pass
                
                if score > 0:
                    candidates.append({
                        "instance_id": workflow["instance_id"],
                        "workflow_name": workflow["workflow_name"],
                        "field": field,
                        "score": score,
                        "signature": signature
                    })
        
        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        best_match = candidates[0] if candidates else None
        best_score = best_match["score"] if best_match else 0.0
        
        logger.info(f"Phase 1 complete: {len(candidates)} candidates, best score: {best_score}")
        
        return {
            "best_score": best_score,
            "best_match": best_match,
            "candidates": candidates
        }
    
    async def _phase2_ocr_analysis(
        self,
        features: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        file_id: str
    ) -> Dict[str, Any]:
        """
        Phase 2: OCR + Keyword analysis.
        
        Returns:
            {
                "best_score": float,
                "best_match": Dict | None,
                "candidates": List[Dict]
            }
        """
        # Check if OCR is already cached in file metadata
        file_meta = file_middleware.get_files_metadata([file_id])
        if not file_meta:
            return {"best_score": 0.0, "best_match": None, "candidates": candidates}
        
        metadata = file_meta[0].get("metadata", {})
        ocr_text = metadata.get("ocr_cache", "")
        
        # If not cached, extract OCR
        if not ocr_text:
            try:
                logger.info(f"🔍 Extracting OCR for file {file_id[:8]}...")
                ocr_text = await extract_text_from_file(file_id)
                
                # Cache OCR result in file metadata
                file_middleware.files_collection.update_one(
                    {"file_id": file_id},
                    {"$set": {"metadata.ocr_cache": ocr_text}}
                )
                logger.info(f"✅ OCR extracted and cached: {len(ocr_text)} chars")
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}")
                return {"best_score": 0.0, "best_match": None, "candidates": candidates}
        
        if not ocr_text:
            return {"best_score": 0.0, "best_match": None, "candidates": candidates}
        
        ocr_text_lower = ocr_text.lower()
        
        # Re-score candidates with OCR
        updated_candidates = []
        
        for candidate in candidates:
            signature = candidate.get("signature", {})
            score = candidate["score"]
            
            # Keyword matching
            keywords = signature.get("keywords", [])
            keyword_matches = sum(1 for kw in keywords if kw.lower() in ocr_text_lower)
            score += keyword_matches * self.KEYWORD_SCORE
            
            # OCR pattern matching
            ocr_patterns = signature.get("ocr_patterns", [])
            pattern_matches = sum(1 for pattern in ocr_patterns if pattern.lower() in ocr_text_lower)
            score += pattern_matches * self.OCR_PATTERN_SCORE
            
            candidate["score"] = score
            updated_candidates.append(candidate)
        
        # Sort by updated score
        updated_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        best_match = updated_candidates[0] if updated_candidates else None
        best_score = best_match["score"] if best_match else 0.0
        
        logger.info(f"Phase 2 complete: best score: {best_score}")
        
        return {
            "best_score": best_score,
            "best_match": best_match,
            "candidates": updated_candidates
        }
    
    async def _phase3_llm_disambiguation(
        self,
        features: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        conversation_context: List[str]
    ) -> Dict[str, Any]:
        """
        Phase 3: LLM-based disambiguation when multiple candidates.
        
        Returns:
            Same format as classify_file()
        """
        try:
            # Build disambiguation prompt
            prompt = self._build_disambiguation_prompt(features, candidates, conversation_context)
            
            # Call LLM
            logger.info("🤖 Calling LLM for file disambiguation...")
            response = self.llm.call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            response_text = response.strip().lower()
            
            # Parse LLM response (expects workflow name or "ambiguous")
            for candidate in candidates:
                if candidate["workflow_name"].lower() in response_text:
                    logger.info(f"✅ LLM matched to: {candidate['workflow_name']}")
                    return {
                        "matched": True,
                        "workflow_instance_id": candidate["instance_id"],
                        "input_field": candidate["field"],
                        "confidence": 0.85,
                        "reasoning": f"LLM context match: {candidate['workflow_name']}",
                        "candidates": []
                    }
            
            # LLM couldn't decide or said ambiguous
            return {
                "matched": False,
                "workflow_instance_id": None,
                "input_field": None,
                "confidence": 0.65,
                "reasoning": "Multiple possible matches. Please clarify which workflow this file is for.",
                "candidates": candidates[:3]
            }
            
        except Exception as e:
            logger.error(f"LLM disambiguation failed: {e}")
            # Fallback to best candidate with medium confidence
            best = candidates[0]
            return {
                "matched": True,
                "workflow_instance_id": best["instance_id"],
                "input_field": best["field"],
                "confidence": 0.70,
                "reasoning": f"Best match (LLM unavailable): {best['workflow_name']}",
                "candidates": []
            }
    
    def _build_disambiguation_prompt(
        self,
        features: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        conversation_context: List[str]
    ) -> str:
        """Build prompt for LLM disambiguation."""
        filename = features["filename"]
        
        context_str = "\n".join(conversation_context[-5:]) if conversation_context else "No recent context"
        
        candidates_str = "\n".join([
            f"- {c['workflow_name']} (waiting for {c['field']})"
            for c in candidates
        ])
        
        prompt = f"""You are helping classify an uploaded file to the correct workflow.

File: {filename}
MIME: {features['mime_type']}

Recent conversation context:
{context_str}

Possible workflows this file could belong to:
{candidates_str}

Based on the filename, file type, and conversation context, which workflow is this file most likely for?
Reply with ONLY the workflow name, or "AMBIGUOUS" if you cannot determine confidently.
"""
        return prompt
