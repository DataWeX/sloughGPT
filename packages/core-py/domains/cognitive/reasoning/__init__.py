"""
Cognitive Reasoning Domain

This module contains the reasoning engine and related components
for advanced logical and creative reasoning.
"""

from .advanced import (
    advanced_reasoning,
    ChainOfThought,
    CausalReasoning,
    ConstitutionalAI,
    ReasoningMode,
    ReasoningResult,
    ReActReasoning,
    SelfConsistency,
    SyllogismReasoning,
    ThoughtStep,
    TreeOfThoughts,
)

from .deep import (
    DeepReasoning,
    DeepReasoningContext,
    FormalLogicEngine,
    LogicalOperator,
    Predicate,
    Term,
    WellFormedFormula,
    WorkingMemory,
    RetrievedKnowledge,
    RetrievalSource,
    Substitution,
)

__all__ = [
    # Basic reasoning
    "ReasoningEngine",
    "advanced_reasoning",
    "ChainOfThought",
    "TreeOfThoughts",
    "SelfConsistency",
    "ConstitutionalAI",
    "CausalReasoning",
    "SyllogismReasoning",
    "ReActReasoning",
    "ReasoningMode",
    "ThoughtStep",
    "ReasoningResult",
    # Deep reasoning
    "DeepReasoning",
    "DeepReasoningContext",
    "RetrievedKnowledge",
    "RetrievalSource",
    # Formal logic
    "FormalLogicEngine",
    "LogicalOperator",
    "Term",
    "Predicate",
    "WellFormedFormula",
    "Substitution",
    # Working memory
    "WorkingMemory",
]


class ReasoningEngine:
    """Advanced reasoning engine with multiple strategies."""

    def __init__(self) -> None:
        self.mode = ReasoningMode.CHAIN_OF_THOUGHT
        self.reasoning_history = []
        self.deep_reasoning = DeepReasoning()
        self.logic_engine = FormalLogicEngine()
        self.working_memory = WorkingMemory()

    async def reason(self, premise: str, context: dict) -> str:
        """Perform reasoning on a premise."""
        result = await advanced_reasoning(
            problem=premise,
            mode=self.mode,
            llm_call=None,
        )
        self.reasoning_history.append(result)
        return result.conclusion

    async def deep_reason(self, problem: str, max_depth: int = 3) -> ReasoningResult:
        """Perform deep reasoning with retrieval and self-correction."""
        return await self.deep_reasoning.reason(problem, max_depth=max_depth)

    async def logical_proof(self, premise1, premise2, conclusion) -> dict:
        """Prove a syllogism using formal logic."""
        return self.logic_engine.prove_syllogism(premise1, premise2, conclusion)

    def assert_fact(self, predicate_name: str, *terms: str) -> None:
        """Assert a fact to the knowledge base."""
        self.logic_engine.assert_predicate(predicate_name, *terms)

    def query(self, predicate_name: str, *terms: str) -> bool:
        """Query the knowledge base."""
        from .deep import Predicate
        return self.logic_engine.query(Predicate(name=predicate_name, terms=[Term(name=t) for t in terms]))

    async def set_mode(self, mode: ReasoningMode) -> None:
        """Set reasoning mode."""
        self.mode = mode

    async def get_history(self) -> list:
        """Get reasoning history."""
        return self.reasoning_history
