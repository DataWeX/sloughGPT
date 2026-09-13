"""Backward-compatibility shim — imports from the new ``domain.cognitive._internal.reasoning`` package."""

from domain.cognitive._internal.reasoning import (
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
from domain.cognitive._internal.reasoning.advanced import (
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
from domain.cognitive._internal.reasoning.deep import (
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
    "DeepReasoning",
    "DeepReasoningContext",
    "RetrievedKnowledge",
    "RetrievalSource",
    "FormalLogicEngine",
    "LogicalOperator",
    "Term",
    "Predicate",
    "WellFormedFormula",
    "Substitution",
    "WorkingMemory",
]


class ReasoningEngine:
    def __init__(self) -> None:
        self.mode = ReasoningMode.CHAIN_OF_THOUGHT
        self.reasoning_history = []
        self.deep_reasoning = DeepReasoning()
        self.logic_engine = FormalLogicEngine()
        self.working_memory = WorkingMemory()

    async def reason(self, premise: str, context: dict) -> str:
        result = await advanced_reasoning(problem=premise, mode=self.mode, llm_call=None)
        self.reasoning_history.append(result)
        return result.conclusion

    async def deep_reason(self, problem: str, max_depth: int = 3) -> ReasoningResult:
        return await self.deep_reasoning.reason(problem, max_depth=max_depth)

    async def logical_proof(self, premise1, premise2, conclusion) -> dict:
        return self.logic_engine.prove_syllogism(premise1, premise2, conclusion)

    def assert_fact(self, predicate_name: str, *terms: str) -> None:
        self.logic_engine.assert_predicate(predicate_name, *terms)

    def query(self, predicate_name: str, *terms: str) -> bool:
        return self.logic_engine.query(Predicate(name=predicate_name, terms=[Term(name=t) for t in terms]))

    async def set_mode(self, mode: ReasoningMode) -> None:
        self.mode = mode

    async def get_history(self) -> list:
        return self.reasoning_history
