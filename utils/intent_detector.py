"""
Execution Intent Detector for the LLM Router.
Detects when user wants to execute a workflow vs just collecting inputs,
and distinguishes questions that need conversational answers.

Uses word-boundary matching for single words to prevent false substring matches
(e.g., "no" matching inside "notes", "ready" matching inside "already").
"""
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def _word_match(keyword: str, text: str) -> bool:
    """Check if keyword appears as a whole word (word-boundary) in text."""
    return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))


class IntentDetector:
    """Detect execution intent from user messages."""
    
    # Exact-match short affirmatives (whole message must be one of these)
    SHORT_AFFIRMATIVES = {
        "y", "yes", "yep", "yeah", "yea", "ok", "okay", "sure",
        "go", "go ahead", "proceed", "confirm", "confirmed",
        "approve", "approved", "alright", "absolutely", "definitely",
        "of course", "let's go", "go for it", "do it", "yes please",
    }
    
    # Exact-match short negatives (whole message must be one of these)
    SHORT_NEGATIVES = {
        "no", "nope", "nah", "stop", "cancel", "abort", "wait",
        "not yet", "hold on", "hold", "pause", "never mind",
    }
    
    # Multi-word delay/cancel phrases (safe for substring match, no false positives)
    DELAY_PHRASES = [
        "hold on", "not yet", "let me check", "let me think",
        "not ready", "don't proceed", "don't do it", "do not proceed",
        "please wait", "i changed my mind",
    ]
    
    # Single-word delay keywords (use word-boundary matching)
    DELAY_WORDS = [
        "cancel", "abort", "pause",
    ]
    
    # Multi-word execute phrases (safe for substring match)
    EXECUTE_PHRASES = [
        "go ahead", "go for it", "let's go", "okay proceed",
        "please proceed", "looks good", "create it", "do it",
        "make it", "process it", "of course",
    ]
    
    # Single-word execute keywords (use word-boundary matching)
    EXECUTE_WORDS = [
        "proceed", "execute", "submit", "confirmed", "confirm",
        "approved", "approve", "accept",
    ]

    # Session/context question patterns (need RAG-style answers)
    SESSION_QUESTION_PATTERNS = [
        "which document", "which file", "what file", "what document",
        "referring to", "are you using", "did you use",
        "what happened", "what was the result", "what did",
        "files uploaded", "files have i", "how many files",
        "what workflows", "previous workflow", "last workflow",
        "what inputs", "what's missing", "what do you need",
        "status of", "current status", "what's the status",
        "what was the po", "what was the invoice", "what number",
        "what amount", "how much", "who was the",
        "what else do you need", "anything else needed",
        "do you need any", "do you need more", "what's left",
        "what remains", "what other",
    ]
    
    # Patterns that look like questions but are actually input/action requests
    FALSE_QUESTION_PATTERNS = [
        "what about", "how about", "what if",
        "can you process", "can you run", "can you execute",
        "can you create", "can you start", "can you send",
        "could you process", "could you run", "could you execute",
        "could you create", "could you start", "could you send",
        "this is the", "here is the", "here's the", "i'm providing",
        "i am providing", "attached is", "please find",
    ]
    
    # Question starters (only match at start of message)
    QUESTION_STARTERS = [
        "what is", "what are", "what was", "what were", "what does", "what did",
        "when ", "where ", "why ", "how ",
        "could you explain", "would you explain", "can you explain",
        "should ", "is it", "are you", "are there",
        "will you",
        "tell me about", "show me", "explain ", "describe ",
    ]
    
    @classmethod
    def detect_execution_intent(cls, message: str) -> str:
        """
        Detect user intent from message.
        
        Returns:
            "execute" - User wants to proceed with workflow
            "delay" - User wants to wait/cancel
            "question" - User is asking a question about session/context
            "collect" - User is providing info for inputs
        """
        if not message:
            return "collect"
        
        msg_lower = message.lower().strip()
        
        # 1. Exact whole-message match for short affirmatives/negatives
        if msg_lower in cls.SHORT_AFFIRMATIVES:
            logger.debug(f"Execute (short affirmative): '{msg_lower}'")
            return "execute"
        
        if msg_lower in cls.SHORT_NEGATIVES:
            logger.debug(f"Delay (short negative): '{msg_lower}'")
            return "delay"
        
        # 2. Multi-word delay phrases (substring safe)
        for phrase in cls.DELAY_PHRASES:
            if phrase in msg_lower:
                logger.debug(f"Delay phrase: '{phrase}'")
                return "delay"
        
        # 3. Single-word delay keywords (word-boundary match)
        for word in cls.DELAY_WORDS:
            if _word_match(word, msg_lower):
                logger.debug(f"Delay word: '{word}'")
                return "delay"
        
        # 4. False-question patterns BEFORE real questions
        #    "This is the delivery note" or "Can you process this?" = input/action
        for pattern in cls.FALSE_QUESTION_PATTERNS:
            if msg_lower.startswith(pattern):
                logger.debug(f"False question (input): '{pattern}'")
                return "collect"
        
        # 5. Session/context questions (before general questions)
        for pattern in cls.SESSION_QUESTION_PATTERNS:
            if pattern in msg_lower:
                logger.debug(f"Session question: '{pattern}'")
                return "question"

        # 6. Question ending with "?"
        if msg_lower.endswith("?"):
            logger.debug("Question detected (ends with ?)")
            return "question"
        
        # 7. Question starters (match at beginning of message)
        for starter in cls.QUESTION_STARTERS:
            if msg_lower.startswith(starter):
                logger.debug(f"Question starter: '{starter}'")
                return "question"
        
        # 8. Multi-word execute phrases (substring safe)
        for phrase in cls.EXECUTE_PHRASES:
            if phrase in msg_lower:
                logger.debug(f"Execute phrase: '{phrase}'")
                return "execute"
        
        # 9. Single-word execute keywords (word-boundary match)
        for word in cls.EXECUTE_WORDS:
            if _word_match(word, msg_lower):
                logger.debug(f"Execute word: '{word}'")
                return "execute"
        
        # 10. Imperative action verbs (word-boundary match)
        action_verbs = ["create", "register", "send", "process", "generate", "upload", "submit"]
        softeners = ["i want", "i need", "i would like", "i'd like", "can you", "could you"]
        for verb in action_verbs:
            if _word_match(verb, msg_lower):
                if not any(s in msg_lower for s in softeners):
                    logger.debug(f"Imperative action: '{verb}'")
                    return "execute"
                break  # Has softener — treat as collect
        
        # Default: collect more info
        logger.debug("No strong intent, defaulting to collect")
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
