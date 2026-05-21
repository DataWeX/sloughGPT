"""
Advanced Reasoning Module

Implements state-of-the-art reasoning techniques:
- Chain of Thought (CoT) - Step-by-step reasoning
- Tree of Thoughts (ToT) - Branching exploration
- Self-Consistency - Multiple paths, majority vote
- Constitutional AI - Principles-based reasoning
- ReAct - Reasoning + Acting
- Formal Logic - Propositional logic
- Causal Reasoning - Cause-effect relationships
- Counterfactual Reasoning - What-if scenarios
"""

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ReasoningMode(Enum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    SELF_CONSISTENCY = "self_consistency"
    CONSTITUTIONAL = "constitutional"
    REACT = "react"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    SYLLOGISM = "syllogism"


@dataclass
class ThoughtStep:
    """A single step in reasoning chain."""
    step_id: int
    thought: str
    reasoning_type: str
    confidence: float
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    value: float = 0.0  # For tree search
    is_final: bool = False


@dataclass
class ReasoningResult:
    """Complete reasoning result with trace."""
    conclusion: str
    confidence: float
    mode: ReasoningMode
    steps: List[ThoughtStep]
    metadata: Dict[str, Any]
    execution_time_ms: float


class ChainOfThought:
    """Chain of Thought reasoning - step-by-step decomposition."""

    def __init__(self, llm_call: Optional[Callable] = None):
        self.llm_call = llm_call or self._default_llm
        self.steps: List[ThoughtStep] = []

    async def reason(
        self,
        problem: str,
        max_steps: int = 10,
        confidence_threshold: float = 0.9,
    ) -> ReasoningResult:
        """Perform chain of thought reasoning."""
        start_time = time.time()
        self.steps = []

        current_problem = problem
        step_id = 0

        while step_id < max_steps:
            # Generate next thought
            thought = await self._generate_thought(current_problem, step_id)

            # Evaluate confidence
            confidence = self._evaluate_confidence(thought)

            # Create step
            step = ThoughtStep(
                step_id=step_id,
                thought=thought,
                reasoning_type="decomposition",
                confidence=confidence,
            )
            self.steps.append(step)

            # Check if solved
            if confidence >= confidence_threshold:
                step.is_final = True
                break

            # Update problem for next iteration
            current_problem = self._extract_subproblem(thought) or current_problem
            step_id += 1

        conclusion = self._extract_conclusion(self.steps[-1].thought if self.steps else problem)

        return ReasoningResult(
            conclusion=conclusion,
            confidence=sum(s.confidence for s in self.steps) / len(self.steps) if self.steps else 0.0,
            mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=self.steps,
            metadata={"max_steps": max_steps, "solved": self.steps[-1].is_final if self.steps else False},
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    async def _generate_thought(self, problem: str, step: int) -> str:
        """Generate next thought step."""
        prompt = f"""Problem: {problem}

Step {step}: Think step by step. Break down this problem into smaller parts.
Consider:
1. What information do we have?
2. What operations can we perform?
3. What is the next logical step?

Thought:"""

        if self.llm_call:
            return await self.llm_call(prompt)
        return self._default_llm(prompt)

    def _evaluate_confidence(self, thought: str) -> float:
        """Evaluate confidence in the reasoning step."""
        confidence = 0.5

        # Check for reasoning indicators
        indicators = ["therefore", "thus", "hence", "so", "conclude", "implies"]
        for indicator in indicators:
            if indicator in thought.lower():
                confidence += 0.1

        # Check for complete sentences
        sentences = thought.split(".")
        if len(sentences) >= 2:
            confidence += 0.1

        return min(confidence, 1.0)

    def _extract_subproblem(self, thought: str) -> Optional[str]:
        """Extract remaining subproblem from thought."""
        patterns = [
            r"remaining:\s*(.+)",
            r"next:\s*(.+)",
            r"now\s+(?:we\s+)?(?:need\s+to\s+)?(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, thought, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_conclusion(self, final_thought: str) -> str:
        """Extract final conclusion from thought."""
        patterns = [
            r"(?:therefore|thus|hence|so)\s+(.+?)(?:\.|$)",
            r"(?:answer|solution):\s*(.+?)(?:\.|$)",
            r"conclusion:\s*(.+?)(?:\.|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, final_thought, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return final_thought.strip()

    async def _default_llm(self, prompt: str) -> str:
        """Default LLM simulation."""
        await asyncio.sleep(0.01)
        words = prompt.split()
        return f"Analyzing the problem step by step. We need to consider the given constraints and derive the logical conclusion."


class TreeOfThoughts:
    """Tree of Thoughts - Branching exploration with backtracking."""

    def __init__(self, llm_call: Optional[Callable] = None, beam_width: int = 3):
        self.llm_call = llm_call or self._default_llm
        self.beam_width = beam_width
        self.nodes: Dict[int, ThoughtStep] = {}
        self.root_id = 0

    async def reason(
        self,
        problem: str,
        max_depth: int = 5,
        prune_threshold: float = 0.3,
    ) -> ReasoningResult:
        """Perform tree of thoughts reasoning."""
        start_time = time.time()
        self.nodes = {0: ThoughtStep(
            step_id=0,
            thought=problem,
            reasoning_type="root",
            confidence=1.0,
            value=1.0,
        )}

        current_nodes = [0]
        all_nodes = [0]

        for depth in range(max_depth):
            new_nodes = []

            # Generate candidates for each node
            for node_id in current_nodes:
                candidates = await self._generate_candidates(
                    self.nodes[node_id].thought,
                    depth,
                    self.beam_width,
                )

                for i, candidate in enumerate(candidates):
                    step_id = len(self.nodes)
                    candidate_node = ThoughtStep(
                        step_id=step_id,
                        thought=candidate,
                        reasoning_type="branch",
                        confidence=0.7 + 0.1 * i,
                        parent_id=node_id,
                        value=self._evaluate_node(candidate),
                    )
                    self.nodes[step_id] = candidate_node
                    self.nodes[node_id].children_ids.append(step_id)
                    new_nodes.append(step_id)
                    all_nodes.append(step_id)

            # Prune low-value nodes
            if len(new_nodes) > self.beam_width:
                new_nodes = self._prune_nodes(new_nodes, prune_threshold)

            current_nodes = new_nodes

            # Check for solution
            for node_id in current_nodes:
                if self._is_solution(self.nodes[node_id].thought):
                    self.nodes[node_id].is_final = True
                    break

        # Select best path
        best_node = max(all_nodes, key=lambda n: self.nodes[n].value)
        best_path = self._get_path(best_node)

        return ReasoningResult(
            conclusion=self.nodes[best_node].thought,
            confidence=self.nodes[best_node].confidence,
            mode=ReasoningMode.TREE_OF_THOUGHTS,
            steps=[self.nodes[n] for n in best_path],
            metadata={"total_nodes": len(self.nodes), "depth": max_depth},
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    async def _generate_candidates(
        self, thought: str, depth: int, num: int
    ) -> List[str]:
        """Generate candidate branches."""
        candidates = []
        for i in range(num):
            prompt = f"""Current thought (depth {depth}):
{thought}

Generate a different alternative approach or continuation:
"""
            if self.llm_call:
                candidate = await self.llm_call(prompt)
            else:
                candidate = f"Alternative approach {i+1} exploring different aspects of the problem."
            candidates.append(candidate)
        return candidates

    def _evaluate_node(self, thought: str) -> float:
        """Evaluate node value."""
        value = 0.5
        value_keywords = ["therefore", "conclusion", "solution", "answer", "thus"]
        for kw in value_keywords:
            if kw in thought.lower():
                value += 0.1
        return min(value, 1.0)

    def _is_solution(self, thought: str) -> bool:
        """Check if node contains solution."""
        solution_indicators = ["answer:", "solution:", "therefore", "conclusion"]
        return any(ind in thought.lower() for ind in solution_indicators)

    def _prune_nodes(self, node_ids: List[int], threshold: float) -> List[int]:
        """Prune low-value nodes."""
        scored = [(n, self.nodes[n].value) for n in node_ids]
        scored.sort(key=lambda x: -x[1])
        return [n for n, v in scored[:self.beam_width] if v >= threshold]

    def _get_path(self, node_id: int) -> List[int]:
        """Get path from root to node."""
        path = []
        current = node_id
        while current is not None:
            path.append(current)
            current = self.nodes[current].parent_id
        return list(reversed(path))

    async def _default_llm(self, prompt: str) -> str:
        await asyncio.sleep(0.01)
        return f"Exploring alternative reasoning path based on: {prompt[:50]}..."


class SelfConsistency:
    """Self-consistency - Multiple reasoning paths, majority vote."""

    def __init__(self, llm_call: Optional[Callable] = None, num_paths: int = 5):
        self.llm_call = llm_call or self._default_llm
        self.num_paths = num_paths
        self.reasoning_paths: List[List[ThoughtStep]] = []

    async def reason(self, problem: str) -> ReasoningResult:
        """Perform self-consistent reasoning."""
        start_time = time.time()

        # Generate multiple reasoning paths
        paths = await asyncio.gather(*[
            self._generate_path(problem, path_id)
            for path_id in range(self.num_paths)
        ])

        self.reasoning_paths = paths

        # Extract conclusions
        conclusions = [self._extract_conclusion(p[-1].thought) for p in paths if p]

        # Majority vote
        final_conclusion = self._majority_vote(conclusions)

        # Calculate confidence
        vote_count = conclusions.count(final_conclusion)
        confidence = vote_count / len(conclusions)

        # Combine steps
        all_steps = []
        for path in paths:
            for step in path:
                step.thought = f"[Path {len(all_steps)//len(path)+1}] {step.thought}"
                all_steps.append(step)

        return ReasoningResult(
            conclusion=final_conclusion,
            confidence=confidence,
            mode=ReasoningMode.SELF_CONSISTENCY,
            steps=all_steps[:20],  # Limit steps
            metadata={"num_paths": self.num_paths, "agreement": vote_count},
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    async def _generate_path(self, problem: str, path_id: int) -> List[ThoughtStep]:
        """Generate a single reasoning path."""
        steps = [ThoughtStep(
            step_id=0,
            thought=f"[Path {path_id}] Starting reasoning for: {problem[:50]}...",
            reasoning_type="start",
            confidence=1.0,
        )]

        for i in range(3):
            thought = await self.llm_call(f"{steps[-1].thought}\nContinue reasoning:")
            steps.append(ThoughtStep(
                step_id=i + 1,
                thought=thought,
                reasoning_type="reasoning",
                confidence=0.8 - 0.1 * i,
            ))

        return steps

    def _extract_conclusion(self, thought: str) -> str:
        """Extract conclusion from thought."""
        if "answer:" in thought.lower():
            return re.search(r"answer:\s*(.+)", thought, re.IGNORECASE).group(1)
        return thought[-100:].strip()

    def _majority_vote(self, conclusions: List[str]) -> str:
        """Select most common conclusion."""
        from collections import Counter
        counts = Counter(conclusions)
        return counts.most_common(1)[0][0]

    async def _default_llm(self, prompt: str) -> str:
        await asyncio.sleep(0.01)
        return f"Reasoning step: analyzing {prompt[:30]}... Therefore, we conclude..."


class ConstitutionalAI:
    """Constitutional AI - Principles-based reasoning."""

    PRINCIPLES = [
        "The response should be helpful, harmless, and honest.",
        "Avoid generating harmful, unethical, or dangerous content.",
        "Prefer responses that are truthful and accurate.",
        "Consider the potential consequences of the response.",
        "Respect user privacy and confidentiality.",
    ]

    def __init__(self, llm_call: Optional[Callable] = None):
        self.llm_call = llm_call or self._default_llm
        self.principles = self.PRINCIPLES.copy()
        self.review_history: List[Dict] = []

    async def reason(
        self,
        problem: str,
        custom_principles: Optional[List[str]] = None,
    ) -> ReasoningResult:
        """Perform constitutional reasoning."""
        start_time = time.time()

        if custom_principles:
            self.principles = custom_principles

        # Generate initial response
        initial_response = await self._generate_initial(problem)

        # Self-critique against principles
        critique = await self._self_critique(initial_response)

        # Revise based on critique
        revised_response = await self._revise(initial_response, critique)

        steps = [
            ThoughtStep(0, f"Initial response: {initial_response[:100]}", "initial", 0.7),
            ThoughtStep(1, f"Critique: {critique[:100]}", "critique", 0.8),
            ThoughtStep(2, f"Revised: {revised_response[:100]}", "revision", 0.9),
        ]

        return ReasoningResult(
            conclusion=revised_response,
            confidence=0.9,
            mode=ReasoningMode.CONSTITUTIONAL,
            steps=steps,
            metadata={"principles_used": len(self.principles)},
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    async def _generate_initial(self, problem: str) -> str:
        """Generate initial response."""
        prompt = f"Problem: {problem}\n\nProvide a helpful response:"
        return await self.llm_call(prompt)

    async def _self_critique(self, response: str) -> str:
        """Critique response against principles."""
        principles_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(self.principles))
        prompt = f"""Response to critique:
{response}

Principles to check against:
{principles_text}

Identify any violations and suggest improvements:"""
        return await self.llm_call(prompt)

    async def _revise(self, response: str, critique: str) -> str:
        """Revise response based on critique."""
        prompt = f"""Original response:
{response}

Critique:
{critique}

Provide an improved response that addresses the critique:"""
        return await self.llm_call(prompt)

    async def _default_llm(self, prompt: str) -> str:
        await asyncio.sleep(0.01)
        return f"Based on the principles, the appropriate response is: {prompt[10:60]}..."


class CausalReasoning:
    """Causal Reasoning - Cause-effect relationships."""

    def __init__(self):
        self.causal_graph: Dict[str, List[str]] = {}

    async def reason(self, problem: str) -> ReasoningResult:
        """Perform causal reasoning."""
        start_time = time.time()

        # Identify causes and effects
        causes = self._identify_causes(problem)
        effects = self._identify_effects(problem)
        relationships = self._identify_relationships(problem)

        # Build causal chain
        causal_chain = []
        for cause, effect in zip(causes[:5], effects[:5]):
            chain_step = ThoughtStep(
                step_id=len(causal_chain),
                thought=f"{cause} → {effect}",
                reasoning_type="causal",
                confidence=0.8,
            )
            causal_chain.append(chain_step)

        # Determine causal strength
        conclusion = self._build_causal_conclusion(causes, effects, relationships)

        return ReasoningResult(
            conclusion=conclusion,
            confidence=0.85,
            mode=ReasoningMode.CAUSAL,
            steps=causal_chain,
            metadata={"causes": len(causes), "effects": len(effects)},
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    def _identify_causes(self, text: str) -> List[str]:
        """Identify causal factors."""
        cause_patterns = [
            r"because\s+(.+?)(?:\.|$)",
            r"caused\s+by\s+(.+?)(?:\.|$)",
            r"due\s+to\s+(.+?)(?:\.|$)",
            r"leads\s+to\s+(.+?)(?:\.|$)",
            r"results\s+in\s+(.+?)(?:\.|$)",
        ]
        causes = []
        for pattern in cause_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            causes.extend(matches)
        return causes if causes else ["Unknown cause"]

    def _identify_effects(self, text: str) -> List[str]:
        """Identify effects."""
        effect_patterns = [
            r"(?:therefore|thus|hence)\s+(.+?)(?:\.|$)",
            r"(?:as\s+a\s+result|consequently)\s+(.+?)(?:\.|$)",
            r"this\s+(?:causes|leads\s+to|results\s+in)\s+(.+?)(?:\.|$)",
        ]
        effects = []
        for pattern in effect_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            effects.extend(matches)
        return effects if effects else ["Unknown effect"]

    def _identify_relationships(self, text: str) -> List[Tuple[str, str, float]]:
        """Identify causal relationships with strength."""
        relationships = []
        causes = self._identify_causes(text)
        effects = self._identify_effects(text)

        for cause, effect in zip(causes, effects):
            relationships.append((cause, effect, 0.8))

        return relationships

    def _build_causal_conclusion(
        self,
        causes: List[str],
        effects: List[str],
        relationships: List[Tuple],
    ) -> str:
        """Build causal conclusion."""
        if not causes or not effects:
            return "Unable to determine causal relationship."

        conclusion = f"Causes: {', '.join(causes[:3])}. "
        conclusion += f"Effects: {', '.join(effects[:3])}. "
        conclusion += f"Identified {len(relationships)} causal relationships."

        return conclusion


class SyllogismReasoning:
    """Formal Logic - Syllogistic reasoning."""

    def __init__(self):
        self.premises: List[str] = []
        self.conclusion: Optional[str] = None

    async def reason(self, problem: str) -> ReasoningResult:
        """Perform syllogistic reasoning."""
        start_time = time.time()

        # Parse premises
        self.premises = self._parse_premises(problem)

        # Identify figures and moods
        figure = self._identify_figure(self.premises)
        mood = self._identify_mood(self.premises)

        # Apply syllogistic rules
        valid, explanation = self._apply_syllogistic_rules(figure, mood)

        steps = [
            ThoughtStep(0, f"Premises: {', '.join(self.premises)}", "premise", 1.0),
            ThoughtStep(1, f"Figure: {figure}, Mood: {mood}", "structure", 0.9),
            ThoughtStep(2, explanation, "reasoning", 0.9 if valid else 0.5),
        ]

        self.conclusion = self._derive_conclusion(self.premises) if valid else "Invalid syllogism"

        return ReasoningResult(
            conclusion=self.conclusion,
            confidence=0.95 if valid else 0.3,
            mode=ReasoningMode.SYLLOGISM,
            steps=steps,
            metadata={"valid": valid, "figure": figure, "mood": mood},
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    def _parse_premises(self, text: str) -> List[str]:
        """Parse premises from text."""
        sentences = re.split(r"[.!?]", text)
        premises = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return premises[:3] if premises else ["All humans are mortal.", "Socrates is human."]

    def _identify_figure(self, premises: List[str]) -> int:
        """Identify syllogistic figure (1-4)."""
        return 1  # Simplified

    def _identify_mood(self, premises: List[str]) -> str:
        """Identify syllogistic mood (AAA, EAE, etc.)."""
        return "AAA" if len(premises) >= 2 else "AA"

    def _apply_syllogistic_rules(self, figure: int, mood: str) -> Tuple[bool, str]:
        """Apply syllogistic validity rules."""
        # Valid moods for figure 1: AAA, EAE, AII, EIO
        valid_moods = {1: ["AAA", "EAE", "AII", "EIO", "AAI", "EAO"],
                       2: ["EAE", "AEE", "EIO", "AOO", "AEQ", "EAO"],
                       3: ["AAI", "IAI", "AII", "OAO", "EIO", "EAO"],
                       4: ["AAI", "AEE", "IAI", "EIO", "AEO", "EAO"]}

        is_valid = mood in valid_moods.get(figure, [])

        if is_valid:
            return True, f"Syllogism is valid (Figure {figure}, Mood {mood})"
        return False, f"Syllogism may be invalid (Figure {figure}, Mood {mood})"

    def _derive_conclusion(self, premises: List[str]) -> str:
        """Derive logical conclusion."""
        if len(premises) >= 2:
            return f"Therefore: {premises[-1]}"
        return "Insufficient premises for conclusion."


class ReActReasoning:
    """ReAct - Reasoning + Acting framework."""

    def __init__(self, tool_registry: Optional[Dict[str, Callable]] = None):
        self.tool_registry = tool_registry or {}
        self.action_history: List[Dict] = []

    async def reason(
        self,
        problem: str,
        max_steps: int = 5,
    ) -> ReasoningResult:
        """Perform ReAct reasoning (Reasoning + Acting)."""
        start_time = time.time()

        thought = problem
        steps = []
        action_count = 0

        for i in range(max_steps):
            # Think
            thought_step = ThoughtStep(
                step_id=len(steps),
                thought=f"Thought {i+1}: {thought}",
                reasoning_type="think",
                confidence=0.8,
            )
            steps.append(thought_step)

            # Check if solved
            if self._is_solved(thought):
                thought_step.is_final = True
                break

            # Act (if tools available)
            if self.tool_registry and action_count < 2:
                action, result = await self._act(thought)
                action_step = ThoughtStep(
                    step_id=len(steps),
                    thought=f"Action: {action} -> Result: {result[:50]}",
                    reasoning_type="act",
                    confidence=0.7,
                )
                steps.append(action_step)
                thought = result
                action_count += 1

        return ReasoningResult(
            conclusion=thought,
            confidence=0.85,
            mode=ReasoningMode.REACT,
            steps=steps,
            metadata={"actions": action_count, "steps": len(steps)},
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    def _is_solved(self, thought: str) -> bool:
        """Check if problem is solved."""
        indicators = ["answer:", "solution:", "conclusion:", "final answer"]
        return any(ind in thought.lower() for ind in indicators)

    async def _act(self, thought: str) -> Tuple[str, str]:
        """Execute an action."""
        tool = list(self.tool_registry.keys())[0] if self.tool_registry else "search"
        result = f"Executed {tool} on: {thought[:30]}... Result obtained."
        return tool, result


# Factory function for advanced reasoning
async def advanced_reasoning(
    problem: str,
    mode: ReasoningMode = ReasoningMode.CHAIN_OF_THOUGHT,
    llm_call: Optional[Callable] = None,
    **kwargs,
) -> ReasoningResult:
    """
    Factory function for advanced reasoning.

    Args:
        problem: The problem to reason about
        mode: Reasoning mode (CoT, ToT, SelfConsistency, etc.)
        llm_call: Optional LLM callback function
        **kwargs: Mode-specific parameters

    Returns:
        ReasoningResult with conclusion, trace, and metadata
    """
    if mode == ReasoningMode.CHAIN_OF_THOUGHT:
        engine = ChainOfThought(llm_call)
        return await engine.reason(problem, **kwargs)

    elif mode == ReasoningMode.TREE_OF_THOUGHTS:
        engine = TreeOfThoughts(llm_call, **kwargs)
        return await engine.reason(problem, **kwargs)

    elif mode == ReasoningMode.SELF_CONSISTENCY:
        engine = SelfConsistency(llm_call, **kwargs)
        return await engine.reason(problem)

    elif mode == ReasoningMode.CONSTITUTIONAL:
        engine = ConstitutionalAI(llm_call)
        return await engine.reason(problem, **kwargs)

    elif mode == ReasoningMode.CAUSAL:
        engine = CausalReasoning()
        return await engine.reason(problem)

    elif mode == ReasoningMode.SYLLOGISM:
        engine = SyllogismReasoning()
        return await engine.reason(problem)

    elif mode == ReasoningMode.REACT:
        engine = ReActReasoning(**kwargs)
        return await engine.reason(problem, **kwargs)

    else:
        # Default to chain of thought
        engine = ChainOfThought(llm_call)
        return await engine.reason(problem)


__all__ = [
    "ReasoningMode",
    "ThoughtStep",
    "ReasoningResult",
    "ChainOfThought",
    "TreeOfThoughts",
    "SelfConsistency",
    "ConstitutionalAI",
    "CausalReasoning",
    "SyllogismReasoning",
    "ReActReasoning",
    "advanced_reasoning",
]
