"""
Execution Intent Detector for the LLM Router.
Detects when user wants to execute a workflow vs just collecting inputs.
"""
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class IntentDetector:
    """Detect execution intent from user messages."""
    
    # Strong execution keywords (high confidence)
    EXECUTE_KEYWORDS = [
        "yes", "proceed", "go ahead", "execute", "run", "start",
        "create it", "do it", "make it", "process it", "submit",
        "confirmed", "confirm", "okay proceed", "let's go", "go for it",
        "please proceed", "ready", "looks good", "looks good proceed",
        "approved", "approve", "accept", "continue",
    ]
    
    # Delay/cancel keywords (user wants to wait)
    DELAY_KEYWORDS = [
        "wait", "hold on", "not yet", "stop", "cancel", "abort",
        "let me check", "let me think", "hold", "pause",
        "not ready", "don't proceed", "don't do it",
    ]
    
    # Question keywords (user asking, not confirming)
    QUESTION_KEYWORDS = [
        "what", "when", "where", "why", "how", "can you",
        "could you", "would you", "should", "is it", "are you",
        "will you", "do you",
    ]
    
    @classmethod
    def detect_execution_intent(cls, message: str) -> str:
        """
        Detect user intent from message.
        
        Returns:
            "execute" - User wants to proceed with workflow
            "delay" - User wants to wait/cancel
            "collect" - User is providing info or asking questions
        """
        if not message:
            return "collect"
        
        msg_lower = message.lower().strip()
        
        # Check for delay/cancel first (highest priority)
        for keyword in cls.DELAY_KEYWORDS:
            if keyword in msg_lower:
                logger.debug(f"🛑 Delay intent detected: '{keyword}' in message")
                return "delay"
        
        # Check for questions (don't auto-execute on questions)
        if msg_lower.endswith("?"):
            logger.debug("❓ Question detected, not executing")
            return "collect"
        
        for keyword in cls.QUESTION_KEYWORDS:
            if msg_lower.startswith(keyword):
                logger.debug(f"❓ Question keyword: '{keyword}'")
                return "collect"
        
        # Check for execution keywords
        for keyword in cls.EXECUTE_KEYWORDS:
            if keyword in msg_lower:
                logger.debug(f"✅ Execute intent detected: '{keyword}' in message")
                return "execute"
        
        # Check for imperative sentences with workflow action verbs
        action_verbs = ["create", "register", "send", "process", "generate", "upload", "submit"]
        if any(verb in msg_lower for verb in action_verbs):
            # "create the po" = execute, "i want to create" = collect
            if not any(phrase in msg_lower for phrase in ["i want", "i need", "can you", "could you"]):
                logger.debug(f"✅ Imperative action detected in message")
                return "execute"
        
        # Default: collect more info
        logger.debug("📝 No strong intent, defaulting to collect")
        return "collect"
    
    @classmethod
    def should_auto_execute(cls, message: str, all_inputs_collected: bool) -> bool:
        """
        Determine if workflow should auto-execute based on message and input status.
        
        Args:
            message: User's message
            all_inputs_collected: Whether all required inputs are ready
        
        Returns:
            True if should execute immediately, False if should ask for confirmation
        """
        if not all_inputs_collected:
            return False
        
        intent = cls.detect_execution_intent(message)
        
        # Only auto-execute if user explicitly wants to proceed
        return intent == "execute"
    
    @classmethod
    def generate_confirmation_prompt(cls, workflow_name: str, inputs_summary: str) -> str:
        """Generate a confirmation prompt when all inputs are ready."""
        return (
            f"✅ All required inputs collected for {workflow_name}!\n\n"
            f"{inputs_summary}\n\n"
            f"Ready to proceed? Reply 'yes' to execute, or provide additional information."
        )
