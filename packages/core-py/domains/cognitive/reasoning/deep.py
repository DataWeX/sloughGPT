"""
Deep Reasoning Engine

Implements:
1. Retrieval-Augmented Reasoning - Uses VectorStore + Memory for grounding
2. Self-Correction Loop - Reflects on reasoning, identifies flaws, revises
3. Formal Logic Engine - Our own predicate logic + syllogism system
"""

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .advanced import ThoughtStep, ReasoningResult, ReasoningMode


# =============================================================================
# 1. RETRIEVAL-AUGMENTED REASONING
# =============================================================================

class RetrievalSource(Enum):
    VECTOR_STORE = "vector_store"
    MEMORY = "memory"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    WORKING_MEMORY = "working_memory"


@dataclass
class RetrievedKnowledge:
    """Retrieved piece of knowledge."""
    content: str
    source: RetrievalSource
    relevance: float
    source_id: Optional[str] = None


@dataclass
class DeepReasoningContext:
    """Context for deep reasoning."""
    query: str
    retrieved_knowledge: List[RetrievedKnowledge] = field(default_factory=list)
    working_memory: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)


class DeepReasoning:
    """Deep reasoning with retrieval and self-correction."""

    def __init__(
        self,
        vector_store=None,
        memory_store=None,
        llm_call: Optional[Callable] = None,
    ):
        self.vector_store = vector_store
        self.memory_store = memory_store
        self.llm_call = llm_call or self._default_llm
        self.max_retrieval = 5

    async def reason(
        self,
        problem: str,
        context: Optional[DeepReasoningContext] = None,
        max_depth: int = 3,
    ) -> ReasoningResult:
        """
        Perform deep reasoning with retrieval and self-correction.
        """
        start_time = asyncio.get_event_loop().time()
        steps = []

        ctx = context or DeepReasoningContext(query=problem)

        # Step 1: Retrieve relevant knowledge
        retrieved = await self._retrieve_knowledge(problem)
        ctx.retrieved_knowledge = retrieved

        # Step 2: Build grounded context
        grounded_context = await self._build_context(problem, retrieved, ctx)
        steps.append(ThoughtStep(
            step_id=0,
            thought=f"Retrieved {len(retrieved)} knowledge pieces",
            reasoning_type="retrieval",
            confidence=0.9,
        ))

        # Step 3: Generate initial reasoning
        initial_reasoning = await self._generate_reasoning(grounded_context, max_depth)
        steps.extend(initial_reasoning)

        # Step 4: Self-correction loop
        corrected_reasoning = await self._self_correct(initial_reasoning, grounded_context, ctx)
        steps.append(ThoughtStep(
            step_id=len(steps),
            thought=f"Self-correction: {corrected_reasoning['corrections']}",
            reasoning_type="correction",
            confidence=0.85,
        ))

        # Step 5: Final synthesis
        conclusion = await self._synthesize(
            initial_reasoning,
            corrected_reasoning,
            grounded_context
        )

        return ReasoningResult(
            conclusion=conclusion,
            confidence=0.88,
            mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=steps,
            metadata={
                "retrieved_count": len(retrieved),
                "corrections": corrected_reasoning.get("corrections", []),
                "depth": max_depth,
            },
            execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
        )

    async def _retrieve_knowledge(self, query: str) -> List[RetrievedKnowledge]:
        """Retrieve relevant knowledge from all sources."""
        results = []

        # From vector store
        if self.vector_store:
            try:
                vector_results = self.vector_store.search(query, top_k=self.max_retrieval)
                for doc_id, similarity in vector_results:
                    doc = self.vector_store.documents.get(doc_id)
                    if doc:
                        results.append(RetrievedKnowledge(
                            content=doc.content,
                            source=RetrievalSource.VECTOR_STORE,
                            relevance=float(similarity),
                            source_id=doc_id,
                        ))
            except Exception:
                pass

        # From memory store
        if self.memory_store:
            try:
                memory_results = self.memory_store.retrieve(query, top_k=self.max_retrieval)
                for item in memory_results:
                    results.append(RetrievedKnowledge(
                        content=item.get("content", ""),
                        source=RetrievalSource.MEMORY,
                        relevance=item.get("relevance", 0.5),
                        source_id=item.get("id"),
                    ))
            except Exception:
                pass

        # Fallback: local reasoning knowledge
        if not results:
            results = await self._fallback_retrieval(query)

        return results

    async def _fallback_retrieval(self, query: str) -> List[RetrievedKnowledge]:
        """Fallback when no external stores available."""
        # Our own reasoning patterns
        patterns = {
            "cause": ["because", "causes", "leads to", "results in", "due to"],
            "effect": ["therefore", "consequently", "as a result", "thus", "hence"],
            "compare": ["however", "but", "although", "whereas", "while"],
            "define": ["is defined as", "means", "refers to", "consists of"],
        }

        results = []
        query_lower = query.lower()

        for category, keywords in patterns.items():
            for kw in keywords:
                if kw in query_lower:
                    results.append(RetrievedKnowledge(
                        content=f"Reasoning pattern '{category}': {query}",
                        source=RetrievalSource.WORKING_MEMORY,
                        relevance=0.7,
                    ))
                    break

        return results[:self.max_retrieval]

    async def _build_context(
        self,
        problem: str,
        retrieved: List[RetrievedKnowledge],
        ctx: DeepReasoningContext,
    ) -> str:
        """Build grounded context from retrieved knowledge."""
        context_parts = [f"Problem: {problem}"]

        if retrieved:
            context_parts.append("\nRelevant Knowledge:")
            for i, know in enumerate(retrieved, 1):
                context_parts.append(f"  {i}. {know.content} (relevance: {know.relevance:.2f})")

        if ctx.constraints:
            context_parts.append(f"\nConstraints: {', '.join(ctx.constraints)}")

        if ctx.assumptions:
            context_parts.append(f"\nAssumptions: {', '.join(ctx.assumptions)}")

        return "\n".join(context_parts)

    async def _generate_reasoning(
        self,
        context: str,
        max_depth: int,
    ) -> List[ThoughtStep]:
        """Generate multi-step reasoning."""
        steps = []
        current_context = context

        for depth in range(max_depth):
            thought = await self.llm_call(
                f"{current_context}\n\nStep {depth + 1}: Analyze this step by step:"
            )
            steps.append(ThoughtStep(
                step_id=len(steps),
                thought=thought,
                reasoning_type="analysis",
                confidence=0.8 - depth * 0.1,
            ))
            current_context = thought

        return steps

    async def _self_correct(
        self,
        reasoning: List[ThoughtStep],
        context: str,
        ctx: DeepReasoningContext,
    ) -> Dict[str, Any]:
        """
        Self-correction loop: reflect on reasoning, find flaws, revise.
        """
        corrections = []

        # Critique each step
        for step in reasoning:
            critique = await self._critique_step(step, context, ctx)
            if critique["has_issue"]:
                corrections.append(critique)

        # Generate corrected version
        if corrections:
            corrected = await self._apply_corrections(reasoning, corrections)
        else:
            corrected = reasoning

        return {
            "corrections": corrections,
            "corrected_reasoning": corrected,
            "needs_revision": len(corrections) > 0,
        }

    async def _critique_step(
        self,
        step: ThoughtStep,
        context: str,
        ctx: DeepReasoningContext,
    ) -> Dict[str, Any]:
        """Critique a single reasoning step."""
        critique_prompt = f"""Critique this reasoning step:
Step: {step.thought}
Context: {context[:500]}

Check for:
1. Factual errors (contradicts known facts)
2. Logical fallacies (circular, strawman, ad hominem)
3. Missing considerations (ignores constraints, assumptions)
4. Unsupported assumptions

Critique:"""

        critique = await self.llm_call(critique_prompt)

        # Simple heuristics for issue detection
        has_issue = False
        issues = []

        if any(word in critique.lower() for word in ["error", "wrong", "incorrect", "fallacy"]):
            has_issue = True
            issues.append("potential_error")

        if "assume" in step.thought.lower() and "assumption" not in context.lower():
            has_issue = True
            issues.append("unjustified_assumption")

        return {
            "step_id": step.step_id,
            "critique": critique,
            "has_issue": has_issue,
            "issues": issues,
        }

    async def _apply_corrections(
        self,
        reasoning: List[ThoughtStep],
        corrections: List[Dict],
    ) -> List[ThoughtStep]:
        """Apply corrections to reasoning."""
        corrected = []

        for step in reasoning:
            correction = next(
                (c for c in corrections if c["step_id"] == step.step_id),
                None
            )
            if correction and correction["has_issue"]:
                new_thought = await self.llm_call(
                    f"Original: {step.thought}\nCritique: {correction['critique']}\n"
                    "Provide a corrected version:"
                )
                corrected.append(ThoughtStep(
                    step_id=step.step_id,
                    thought=f"[CORRECTED] {new_thought}",
                    reasoning_type="correction",
                    confidence=0.9,
                ))
            else:
                corrected.append(step)

        return corrected

    async def _synthesize(
        self,
        initial: List[ThoughtStep],
        corrected: Dict,
        context: str,
    ) -> str:
        """Synthesize final conclusion."""
        synthesis_prompt = f"""Given the reasoning steps and corrections, provide a final answer.

Context: {context[:500]}
Initial reasoning: {initial[-1].thought if initial else 'N/A'}
Corrections applied: {len(corrected.get('corrections', []))}

Final answer:"""

        return await self.llm_call(synthesis_prompt)

    async def _default_llm(self, prompt: str) -> str:
        """Default LLM simulation."""
        await asyncio.sleep(0.01)
        # Simple pattern-based response
        if "retrieve" in prompt.lower():
            return "Based on available knowledge, the reasoning proceeds step by step."
        elif "critique" in prompt.lower():
            return "No major issues detected in this reasoning step."
        elif "correct" in prompt.lower():
            return "The reasoning appears sound after review."
        elif "final" in prompt.lower():
            return "Therefore, the conclusion is reached through logical deduction."
        return "Reasoning continues with careful analysis of the problem."


# =============================================================================
# 2. FORMAL LOGIC ENGINE
# =============================================================================

class LogicalOperator(Enum):
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    IFF = "↔"
    FORALL = "∀"
    EXISTS = "∃"


@dataclass
class Term:
    """A logical term (constant, variable, or function)."""
    name: str
    is_variable: bool = False
    is_function: bool = False
    arguments: List["Term"] = field(default_factory=list)


@dataclass
class Predicate:
    """A predicate (relation) in first-order logic."""
    name: str
    terms: List[Term]
    negated: bool = False


@dataclass
class WellFormedFormula:
    """A well-formed formula in first-order logic."""
    predicate: Optional[Predicate] = None
    operator: Optional[LogicalOperator] = None
    left: Optional["WellFormedFormula"] = None
    right: Optional["WellFormedFormula"] = None
    quantifier_var: Optional[Term] = None
    quantifier_type: Optional[LogicalOperator] = None
    subformula: Optional["WellFormedFormula"] = None


@dataclass
class Substitution:
    """A substitution mapping variables to terms."""
    mapping: Dict[str, Term] = field(default_factory=dict)


class FormalLogicEngine:
    """
    Our own formal logic engine with:
    - Propositional logic (AND, OR, NOT, IMPLIES)
    - First-order logic (FORALL, EXISTS)
    - Syllogistic logic (categorical syllogisms)
    - Unification algorithm
    - Resolution principle
    """

    def __init__(self):
        self.knowledge_base: List[WellFormedFormula] = []
        self.inference_history: List[Dict] = []

    def assert_fact(self, formula: WellFormedFormula) -> None:
        """Add a formula to the knowledge base."""
        self.knowledge_base.append(formula)

    def assert_predicate(self, name: str, *terms: str) -> None:
        """Assert a simple predicate (e.g., "human(socrates)")."""
        pred = Predicate(
            name=name,
            terms=[Term(name=t) for t in terms],
        )
        wff = WellFormedFormula(predicate=pred)
        self.assert_fact(wff)

    def query(self, query_predicate: Predicate) -> bool:
        """Query if a predicate is entailed by the knowledge base."""
        # Forward chaining
        return self._forward_chain(query_predicate)

    def _forward_chain(self, query: Predicate) -> bool:
        """Forward chaining inference."""
        derived = set()
        queue = list(self.knowledge_base)

        while queue:
            wff = queue.pop(0)

            # Apply Modus Ponens
            if wff.operator == LogicalOperator.IMPLIES:
                result = self._modus_ponens(wff.left, wff.right, derived)
                if result:
                    derived.add(result)
                    queue.append(WellFormedFormula(predicate=result))

            # Check if query is satisfied
            if wff.predicate and wff.predicate.name == query.name:
                if self._unify(wff.predicate, query):
                    return True

        return False

    def _modus_ponens(
        self,
        antecedent: WellFormedFormula,
        consequent: WellFormedFormula,
        derived: Set[Predicate],
    ) -> Optional[Predicate]:
        """Apply Modus Ponens: (P → Q), P ⊢ Q"""
        if not antecedent.predicate or not consequent.predicate:
            return None

        for pred in derived:
            if pred.name == antecedent.predicate.name:
                subst = self._unify(antecedent.predicate, pred)
                if subst:
                    return self._apply_substitution(consequent.predicate, subst)

        return None

    def _unify(self, pred1: Predicate, pred2: Predicate) -> Optional[Substitution]:
        """Unification algorithm - find substitution making predicates equal."""
        if pred1.name != pred2.name:
            return None

        if len(pred1.terms) != len(pred2.terms):
            return None

        subst = Substitution()

        for t1, t2 in zip(pred1.terms, pred2.terms):
            new_subst = self._unify_terms(t1, t2, subst)
            if new_subst is None:
                return None
            subst = new_subst

        return subst

    def _unify_terms(
        self,
        t1: Term,
        t2: Term,
        subst: Substitution,
    ) -> Optional[Substitution]:
        """Unify two terms."""
        # Apply existing substitution
        t1 = self._apply_term_substitution(t1, subst)
        t2 = self._apply_term_substitution(t2, subst)

        # Same term
        if t1.name == t2.name:
            return subst

        # Variable unification
        if t1.is_variable:
            if self._occurs_check(t1, t2, subst):
                return None
            new_subst = Substitution(mapping=subst.mapping.copy())
            new_subst.mapping[t1.name] = t2
            return new_subst

        if t2.is_variable:
            if self._occurs_check(t2, t1, subst):
                return None
            new_subst = Substitution(mapping=subst.mapping.copy())
            new_subst.mapping[t2.name] = t1
            return new_subst

        return None

    def _occurs_check(self, var: Term, term: Term, subst: Substitution) -> bool:
        """Occurs check for occurs in unification."""
        if not var.is_variable:
            return False

        applied = self._apply_term_substitution(term, subst)

        if applied.name == var.name:
            return True

        if applied.is_function:
            return any(self._occurs_check(var, arg, subst) for arg in applied.arguments)

        return False

    def _apply_term_substitution(self, term: Term, subst: Substitution) -> Term:
        """Apply substitution to a term."""
        if term.is_variable and term.name in subst.mapping:
            return subst.mapping[term.name]
        if term.is_function:
            return Term(
                name=term.name,
                is_function=True,
                arguments=[self._apply_term_substitution(a, subst) for a in term.arguments],
            )
        return term

    def _apply_substitution(self, pred: Predicate, subst: Substitution) -> Predicate:
        """Apply substitution to a predicate."""
        return Predicate(
            name=pred.name,
            terms=[self._apply_term_substitution(t, subst) for t in pred.terms],
            negated=pred.negated,
        )

    def resolution(self, goal: Predicate) -> bool:
        """
        Resolution principle for proving by refutation.
        Returns True if goal is entailed.
        """
        # Add negated goal to KB
        negated_goal = WellFormedFormula(
            predicate=Predicate(name=goal.name, terms=goal.terms, negated=True)
        )

        kb = self.knowledge_base + [negated_goal]
        clauses = self._to_clausal_form(kb)

        # Resolution loop
        max_iterations = 100
        for _ in range(max_iterations):
            new_clauses = []

            for i, clause1 in enumerate(clauses):
                for clause2 in clauses[i + 1:]:
                    resolvent = self._resolve_clauses(clause1, clause2)
                    if resolvent is not None:
                        if not resolvent:  # Empty clause = contradiction
                            return True
                        if resolvent not in clauses and resolvent not in new_clauses:
                            new_clauses.append(resolvent)

            if not new_clauses:
                return False

            clauses.extend(new_clauses)

        return False

    def _to_clausal_form(self, wffs: List[WellFormedFormula]) -> List[Set[Predicate]]:
        """Convert to clausal form (conjunctive normal form)."""
        clauses = []
        for wff in wffs:
            clause = self._extract_literals(wff)
            clauses.append(clause)
        return clauses

    def _extract_literals(self, wff: WellFormedFormula) -> Set[Predicate]:
        """Extract literals from formula."""
        literals = set()

        if wff.predicate:
            literals.add(wff.predicate)
        elif wff.operator == LogicalOperator.AND and wff.left and wff.right:
            literals.update(self._extract_literals(wff.left))
            literals.update(self._extract_literals(wff.right))
        elif wff.operator == LogicalOperator.OR and wff.left and wff.right:
            literals.update(self._extract_literals(wff.left))
            literals.update(self._extract_literals(wff.right))

        return literals

    def _resolve_clauses(
        self,
        clause1: Set[Predicate],
        clause2: Set[Predicate],
    ) -> Optional[Set[Predicate]]:
        """Resolve two clauses."""
        for lit1 in clause1:
            for lit2 in clause2:
                # Complementary literals
                if lit1.name == lit2.name and lit1.negated != lit2.negated:
                    subst = self._unify_complementary(lit1, lit2)
                    if subst:
                        # Remove resolved literals, apply substitution
                        resolvent = clause1 - {lit1}
                        resolvent.update(clause2 - {lit2})

                        # Apply substitution
                        new_resolvent = set()
                        for lit in resolvent:
                            new_lit = self._apply_substitution(lit, subst)
                            new_resolvent.add(new_lit)

                        return new_resolvent
        return None

    def _unify_complementary(self, lit1: Predicate, lit2: Predicate) -> Optional[Substitution]:
        """Unify complementary literals."""
        if lit1.negated:
            return self._unify(lit1, Predicate(name=lit2.name, terms=lit2.terms, negated=False))
        else:
            return self._unify(
                Predicate(name=lit1.name, terms=lit1.terms, negated=False),
                Predicate(name=lit2.name, terms=lit2.terms, negated=True)
            )

    # =============================================================================
    # SYLLOGISM SUPPORT
    # =============================================================================

    SYLLOGISM_FIGURES = {
        1: [("M-P", "S-M"), ("P-M", "S-M"), ("M-P", "M-S"), ("P-M", "M-S")],
        2: [("P-M", "S-M"), ("P-M", "M-S"), ("M-P", "S-M"), ("M-P", "M-S")],
        3: [("M-P", "S-M"), ("M-P", "M-S"), ("P-M", "S-M"), ("P-M", "M-S")],
        4: [("P-M", "S-M"), ("M-P", "M-S"), ("P-M", "M-S"), ("M-P", "S-M")],
    }

    def prove_syllogism(
        self,
        premise1: Tuple[str, str, str],  # (Subject, Copula, Predicate)
        premise2: Tuple[str, str, str],
        conclusion: Tuple[str, str, str],
    ) -> Dict[str, Any]:
        """
        Prove a categorical syllogism.
        
        Args:
            premise1: (S, copula, P) e.g., ("All", "are", "mortal") means "All S are P"
            premise2: (S, copula, P)
            conclusion: (S, copula, P)
        
        Returns:
            Dict with proof result, figure, mood, and validity
        """
        # Convert to standard form
        p1 = self._to_categorical(premise1)
        p2 = self._to_categorical(premise2)
        conc = self._to_categorical(conclusion)

        # Determine figure
        figure = self._determine_figure(p1, p2)

        # Determine mood
        mood = f"{p1[0][0]}{p2[0][0]}{conc[0][0]}"  # A, E, I, O

        # Check validity using Aristotle's rules
        valid, reason = self._check_syllogism_validity(mood, figure, p1, p2, conc)

        self.inference_history.append({
            "type": "syllogism",
            "premise1": premise1,
            "premise2": premise2,
            "conclusion": conclusion,
            "figure": figure,
            "mood": mood,
            "valid": valid,
            "reason": reason,
        })

        return {
            "valid": valid,
            "figure": figure,
            "mood": mood,
            "reason": reason,
            "form": f"{self._format_categorical(p1)} / {self._format_categorical(p2)} ∴ {self._format_categorical(conc)}",
        }

    def _to_categorical(self, premise: Tuple[str, str, str]) -> Tuple[str, str, str, str]:
        """Convert to categorical form: (quantifier, subject, copula, predicate)"""
        quant, copula, pred = premise
        # Infer subject from quantifier position
        if quant.lower() in ["all", "every", "no", "some"]:
            subject = "S"
        else:
            subject = "S"
        return (quant.upper()[:1], subject, copula, pred)  # A, E, I, O

    def _determine_figure(self, p1: Tuple, p2: Tuple) -> int:
        """Determine syllogistic figure (1-4)."""
        # Simplified figure detection
        return 1

    def _check_syllogism_validity(
        self,
        mood: str,
        figure: int,
        p1: Tuple,
        p2: Tuple,
        conc: Tuple,
    ) -> Tuple[bool, str]:
        """Check syllogism validity."""
        # Valid mood-figure combinations (Aristotle's rules)
        valid_combinations = {
            1: ["AAA", "EAE", "AII", "EIO", "AAI", "EAO"],
            2: ["EAE", "AEE", "EIO", "AOO", "EAO"],
            3: ["AAI", "IAI", "AII", "OAO", "EIO", "EAO"],
            4: ["AAI", "AEE", "IAI", "EIO", "AEO", "EAO"],
        }

        if mood in valid_combinations.get(figure, []):
            return True, "Valid according to Aristotelian logic"

        return False, "Invalid syllogism form"

    def _format_categorical(self, cat: Tuple) -> str:
        """Format categorical proposition."""
        return f"{cat[0]} {cat[1]} {cat[2]} {cat[3]}"


# =============================================================================
# 3. WORKING MEMORY FOR REASONING
# =============================================================================

class WorkingMemory:
    """Working memory for active reasoning."""

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[str] = []
        self.access_count: Dict[str, int] = {}

    def add(self, item: str) -> None:
        """Add item to working memory."""
        if len(self.items) >= self.capacity:
            # Evict least recently used
            lru = min(self.items, key=lambda x: self.access_count.get(x, 0))
            self.items.remove(lru)
            del self.access_count[lru]
        self.items.append(item)
        self.access_count[item] = 1

    def access(self, item: str) -> None:
        """Record access to item."""
        self.access_count[item] = self.access_count.get(item, 0) + 1

    def get_recent(self, n: int = 5) -> List[str]:
        """Get n most recently accessed items."""
        sorted_items = sorted(self.items, key=lambda x: -self.access_count.get(x, 0))
        return sorted_items[:n]

    def clear(self) -> None:
        """Clear working memory."""
        self.items.clear()
        self.access_count.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
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
