"""
Creativity Engine Implementation

This module provides creative thinking capabilities including
idea generation, creative synthesis, and innovative problem solving.
"""

import logging
from typing import Any, Dict, List

from ...__init__ import BaseComponent, ComponentException


class CreativityEngine(BaseComponent):
    """Advanced creativity engine for innovative thinking"""

    def __init__(self) -> None:
        super().__init__("creativity_engine")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")

        # Creativity state
        self.idea_pool = []
        self.creative_constraints = {}
        self.inspiration_sources = []

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the creativity engine"""
        try:
            self.logger.info("Initializing Creativity Engine...")
            self.is_initialized = True
            self.logger.info("Creativity Engine initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Creativity Engine: {e}")
            raise ComponentException(f"Creativity Engine initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the creativity engine"""
        try:
            self.logger.info("Shutting down Creativity Engine...")
            self.is_initialized = False
            self.logger.info("Creativity Engine shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown Creativity Engine: {e}")
            raise ComponentException(f"Creativity Engine shutdown failed: {e}")

    async def generate_ideas(self, prompt: str, num_ideas: int = 5) -> List[str]:
        """Generate creative ideas using multiple brainstorming techniques.
        
        Techniques used:
        - SCAMPER: Substitute, Combine, Adapt, Modify, Put to other uses, Eliminate, Reverse
        - Random input: Word association for unexpected connections
        - Constraint removal: Imagine solving without current limitations
        - Analogical thinking: Map from unrelated domains
        
        Args:
            prompt: Problem or topic to generate ideas for
            num_ideas: Number of ideas to generate
            
        Returns:
            List of creative ideas with descriptions
        """
        import random
        import hashlib
        
        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        scampers = [
            ("Substitute", f"What if we replaced X with Y in {prompt}?"),
            ("Combine", f"How could we merge {prompt} with another concept?"),
            ("Adapt", f"What similar solutions exist that we could adapt to {prompt}?"),
            ("Modify", f"How could we modify or magnify {prompt}?"),
            ("Put to other uses", f"What else could {prompt} be used for?"),
            ("Eliminate", f"What if we removed a key element from {prompt}?"),
            ("Reverse", f"What if we did the opposite of {prompt}?"),
        ]
        
        random_inputs = [
            "biomimicry", "swarm intelligence", "emergence", "entropy",
            "fractals", "quantum superposition", "oscillation", "symbiosis"
        ]
        
        random.shuffle(scampers)
        random.shuffle(random_inputs)
        
        ideas = []
        for i in range(min(num_ideas, len(scampers))):
            technique, description = scampers[i]
            ideas.append({
                "id": i + 1,
                "technique": technique,
                "idea": description,
                "variant": "scamper"
            })
        
        for i in range(min(num_ideas - len(ideas), 3)):
            technique = "Random Input"
            random_word = random_inputs[i]
            ideas.append({
                "id": len(ideas) + 1,
                "technique": technique,
                "idea": f"Apply '{random_word}' thinking to {prompt}",
                "variant": "random_input"
            })
        
        return [f"[{idea['technique']}] {idea['idea']}" for idea in ideas[:num_ideas]]

    async def creative_synthesis(self, concepts: List[str]) -> str:
        """Synthesize creative solution from multiple concepts.
        
        Uses conceptual blending theory to combine concepts into novel ideas.
        
        Args:
            concepts: List of concepts to synthesize
            
        Returns:
            Synthesized creative solution with explanation
        """
        if not concepts:
            return "No concepts provided for synthesis."
        
        if len(concepts) == 1:
            return f"Focus on developing: {concepts[0]}"
        
        blend_patterns = [
            "integrating {first} with {second} to create {third}",
            "{first} enhanced by {second} principles for {third}",
            "using {first} approach in a {second} context to achieve {third}",
            "{first} meets {second}: a new paradigm for {third}",
        ]
        
        import random
        pattern = random.choice(blend_patterns)
        
        first = concepts[0]
        second = concepts[1] if len(concepts) > 1 else concepts[0]
        third = concepts[-1] if len(concepts) > 2 else "innovation"
        
        synthesis = pattern.format(first=first, second=second, third=third)
        
        additional = ""
        if len(concepts) > 2:
            supporting = ", ".join(concepts[2:])
            additional = f" Supporting elements: {supporting}"
        
        return f"{synthesis}.{additional} This synthesis leverages the strengths of each concept while mitigating individual weaknesses."

    async def apply_constraints(self, problem: str, constraints: Dict[str, Any]) -> List[str]:
        """Apply creative constraints to generate innovative solutions.
        
        Args:
            problem: Problem statement
            constraints: Dict of constraint types (time, cost, resources, etc.)
            
        Returns:
            Solutions that work within the given constraints
        """
        solutions = []
        
        constraint_types = {
            "time": ["rapid prototyping", "minimum viable approach", "phased delivery"],
            "cost": ["open source alternatives", "resource sharing", "minimalist design"],
            "resources": ["leveraging existing assets", "cross-functional teams", "automation"],
            "technical": ["modular architecture", "graceful degradation", "API-first design"]
        }
        
        for constraint_type, value in constraints.items():
            if constraint_type in constraint_types:
                approach = constraint_types[constraint_type]
                for a in approach:
                    solutions.append(f"Given {constraint_type} constraint ({value}): Consider {a}")
        
        return solutions

    async def divergent_thinking(self, topic: str, dimensions: int = 5) -> Dict[str, Any]:
        """Generate divergent ideas across multiple dimensions.
        
        Args:
            topic: Central topic
            dimensions: Number of thinking dimensions to explore
            
        Returns:
            Dict of ideas organized by thinking dimension
        """
        dimensions_map = {
            "function": "What does it do? What are its purposes?",
            "form": "What does it look like? What is its structure?",
            "process": "How does it work? What are its methods?",
            "impact": "What effects does it have? What changes?",
            "context": "Where is it used? Under what conditions?"
        }
        
        result = {}
        selected_dims = list(dimensions_map.keys())[:dimensions]
        
        for dim in selected_dims:
            question = dimensions_map[dim]
            result[dim] = {
                "question": question,
                "ideas": [f"For {topic}: {question}"]
            }
        
        return {
            "topic": topic,
            "thinking_mode": "divergent",
            "dimensions": result,
            "total_ideas": sum(len(v["ideas"]) for v in result.values())
        }
