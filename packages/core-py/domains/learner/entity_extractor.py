"""
Entity and relationship extractor for automatic knowledge graph population.

Extracts entities and relationships from chat conversations and stores
them as knowledge facts. Facts are then auto-retrieved during future
conversations via ``_build_full_prompt()``.

Patterns matched:
- "X is a|an Y" → type relationship
- "X likes|has|wants|uses Y" → action relationship
- "X's Y" → possession
- Capitalized multi-word phrases → named entities
- Repeated nouns across messages → significant entities
"""

import re
import logging
from typing import List, Set, Tuple
from collections import defaultdict

logger = logging.getLogger("man.learner.entity_extractor")

# Relationship patterns: (regex, relationship_type)
_RELATION_PATTERNS = [
    (re.compile(r'(\w+(?:\s+\w+){0,3})\s+is\s+(?:a|an|the)\s+(\w+(?:\s+\w+){0,3})', re.I), "is_a"),
    (re.compile(r'(\w+(?:\s+\w+){0,3})\s+likes\s+(\w+(?:\s+\w+){0,3})', re.I), "likes"),
    (re.compile(r'(\w+(?:\s+\w+){0,3})\s+has\s+(\w+(?:\s+\w+){0,3})', re.I), "has"),
    (re.compile(r'(\w+(?:\s+\w+){0,3})\s+wants\s+(\w+(?:\s+\w+){0,3})', re.I), "wants"),
    (re.compile(r'(\w+(?:\s+\w+){0,3})\s+uses\s+(\w+(?:\s+\w+){0,3})', re.I), "uses"),
    (re.compile(r'(\w+(?:\s+\w+){0,3})\s+works\s+at\s+(\w+(?:\s+\w+){0,3})', re.I), "works_at"),
    (re.compile(r'(\w+(?:\s+\w+){0,3})\s+lives\s+in\s+(\w+(?:\s+\w+){0,3})', re.I), "lives_in"),
    (re.compile(r'(\w+(?:\s+\w+){0,3})\s+created\s+(\w+(?:\s+\w+){0,3})', re.I), "created"),
    (re.compile(r'(\w+(?:\s+\w+){0,3})\s+called\s+(\w+(?:\s+\w+){0,3})', re.I), "called"),
    (re.compile(r"(\w+)'s\s+(\w+(?:\s+\w+){0,3})", re.I), "possesses"),
]

_STOP_WORDS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "shall", "should", "may", "might",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "theirs",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "up", "out", "also", "well", "here", "there",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "get", "got", "make", "made", "take", "took", "know", "think",
    "see", "want", "give", "tell", "come", "go", "look", "use", "find",
}


def _is_valid_entity(word: str) -> bool:
    """Check if a word is a plausible entity (not a stop word, not punctuation)."""
    return bool(re.match(r'^[A-Za-z][a-zA-Z\']{1,}$', word)) and word.lower() not in _STOP_WORDS


_COMMON_FALSE_ENTITIES = {"Nice", "Hello", "Hi", "Hey", "Thanks", "Please", "Sure",
                          "Yes", "No", "Okay", "Ok", "Great", "Good", "Right", "Well",
                          "So", "Also", "Here", "There", "Really", "Actually", "Just"}


def extract_entities(text: str) -> List[str]:
    """Extract named entities from text using heuristics.

    Captures capitalized multi-word sequences and repeated significant nouns.
    Returns deduplicated list. Single words that are part of a multi-word entity
    are excluded to avoid duplicates.
    """
    sentences = re.split(r'[.!?]+', text)
    entities: List[str] = []
    multi_word: Set[str] = set()

    for sentence in sentences:
        caps_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', sentence)
        for m in caps_matches:
            clean = m.strip()
            if clean and clean not in multi_word and clean not in _COMMON_FALSE_ENTITIES:
                entities.append(clean)
                multi_word.add(clean)
                for part in clean.split():
                    multi_word.add(part)

        for match in re.finditer(r'\b([A-Z][a-z]{2,})\b', sentence):
            w = match.group(1)
            if w not in multi_word and w not in _COMMON_FALSE_ENTITIES and _is_valid_entity(w):
                entities.append(w)
                multi_word.add(w)

    return entities


def extract_relationships(text: str) -> List[Tuple[str, str, str]]:
    """Extract (subject, relationship, object) triples from text."""
    relationships: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for pattern, rel_type in _RELATION_PATTERNS:
        for match in pattern.finditer(text):
            subj = match.group(1).strip()
            obj = match.group(2).strip()
            obj = re.sub(r'^(a|an|the)\s+', '', obj, flags=re.I)
            triple = (subj, rel_type, obj)
            if triple not in seen and subj.lower() not in _STOP_WORDS and len(obj) > 1:
                relationships.append(triple)
                seen.add(triple)

    return relationships


def extract_facts_from_conversation(user_msg: str, assistant_msg: str) -> List[str]:
    """Extract knowledge facts from a chat exchange.

    Returns natural-language fact strings ready for storage in KnowledgeMemory.
    """
    facts: List[str] = []
    seen_facts: Set[str] = set()

    combined = f"{user_msg} {assistant_msg}"

    # Extract relationships
    for subj, rel, obj in extract_relationships(combined):
        if rel == "is_a":
            fact = f"{subj} is a {obj}"
        elif rel == "likes":
            fact = f"{subj} likes {obj}"
        elif rel == "has":
            fact = f"{subj} has {obj}"
        elif rel == "possesses":
            fact = f"{subj} has {obj}"
        else:
            fact = f"{subj} {rel} {obj}"

        if fact not in seen_facts and len(fact) > 5:
            facts.append(fact)
            seen_facts.add(fact)

    # Extract named entities as standalone facts
    entities = extract_entities(combined)
    for ent in entities[:5]:
        if ent.lower() in _STOP_WORDS or ent in _COMMON_FALSE_ENTITIES:
            continue
        fact = f"Entity {ent} exists"
        if fact not in seen_facts:
            facts.append(fact)
            seen_facts.add(fact)

    return facts


async def extract_facts_neural(user_msg: str, assistant_msg: str) -> List[str]:
    """Use the current LLM to extract nuanced facts from a conversation.

    Returns a list of natural-language facts.
    """
    try:
        from domains.infrastructure.model_server import get_model_registry
        registry = get_model_registry()
        if not registry or not registry.list_models():
            return []

        model = registry.get_default_model()
        if not model:
            return []

        combined = f"User: {user_msg}\nAI: {assistant_msg}"
        prompt = (
            "Extract a few concise, factual statements about the user or mentioned entities "
            "from this conversation. Focus on new knowledge. Return only a bulleted list of facts, "
            "one per line, without labels or introductory text.\n\n"
            f"Conversation:\n{combined}\n\n"
            "Facts:"
        )

        # Use non-streaming generate for extraction
        result = await model.generate(prompt, max_new_tokens=128, temperature=0.1)
        text = result.text.strip()
        if not text:
            return []

        facts = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
        return [f for f in facts if len(f) > 5]
    except Exception as e:
        logger.debug(f"Neural extraction failed: {e}")
        return []


async def extract_and_store(user_msg: str, assistant_msg: str, knowledge_memory=None):
    """Extract facts from conversation and store in KnowledgeMemory.

    Args:
        user_msg: User's message text
        assistant_msg: Assistant's response text
        knowledge_memory: KnowledgeMemory instance (defaults to global singleton)

    Returns:
        Number of new facts stored
    """
    try:
        # 1. Rule-based extraction (Fast, always runs)
        facts = extract_facts_from_conversation(user_msg, assistant_msg)

        # 2. Neural extraction (Slower, runs for significant exchanges)
        # Significant exchange: total length > 100 chars or contains named entities
        if len(user_msg) + len(assistant_msg) > 100:
            neural_facts = await extract_facts_neural(user_msg, assistant_msg)
            facts.extend(neural_facts)

        if not facts:
            return 0

        from domains.learner.knowledge import KnowledgeFact
        if knowledge_memory is None:
            from domains.learner.knowledge import get_knowledge_memory
            knowledge_memory = get_knowledge_memory()

        stored = 0
        for fact_text in facts:
            try:
                fact = KnowledgeFact(
                    content=fact_text,
                    topic="chat",
                    source="auto_extracted",
                )
                if knowledge_memory.add_fact(fact):
                    stored += 1
            except Exception:
                continue

        if stored > 0:
            logger.info(f"Auto-extracted {stored} facts from conversation")

        return stored
    except Exception as e:
        logger.debug(f"Entity extraction skipped: {e}")
        return 0
