"""
Learning Optimizer Implementation

This module provides learning optimization capabilities including
spaced repetition, adaptive learning, and knowledge consolidation.
"""

import logging
from typing import Any, Dict, List

from ...__init__ import BaseComponent, ComponentException


class LearningOptimizer(BaseComponent):
    """Advanced learning optimization system"""

    def __init__(self) -> None:
        super().__init__("learning_optimizer")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")

        # Learning state
        self.learning_sessions = []
        self.spaced_repetition_schedule = {}
        self.adaptive_learning_params = {}

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the learning optimizer"""
        try:
            self.logger.info("Initializing Learning Optimizer...")
            self.is_initialized = True
            self.logger.info("Learning Optimizer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Learning Optimizer: {e}")
            raise ComponentException(f"Learning Optimizer initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the learning optimizer"""
        try:
            self.logger.info("Shutting down Learning Optimizer...")
            self.is_initialized = False
            self.logger.info("Learning Optimizer shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown Learning Optimizer: {e}")
            raise ComponentException(f"Learning Optimizer shutdown failed: {e}")

    async def optimize_learning_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a learning session using adaptive learning principles.
        
        Args:
            session_data: Dict containing session_id, user_id, content, difficulty, 
                         performance_history, time_spent
            
        Returns:
            Optimized session parameters and recommendations
        """
        session_id = session_data.get("session_id", "unknown")
        difficulty = session_data.get("difficulty", 0.5)
        performance_history = session_data.get("performance_history", [])
        time_spent = session_data.get("time_spent", 0)
        
        avg_performance = sum(performance_history) / len(performance_history) if performance_history else 0.5
        
        if avg_performance > 0.85:
            next_difficulty = min(1.0, difficulty + 0.1)
            recommendation = "increase_difficulty"
        elif avg_performance < 0.6:
            next_difficulty = max(0.1, difficulty - 0.1)
            recommendation = "decrease_difficulty"
        else:
            next_difficulty = difficulty
            recommendation = "maintain"
        
        optimal_session_length = max(15, min(45, 30 - (time_spent / 60) * 10))
        
        return {
            "optimized": True,
            "session_id": session_id,
            "recommended_difficulty": next_difficulty,
            "recommendation": recommendation,
            "optimal_session_length_minutes": optimal_session_length,
            "break_interval_minutes": 15 if time_spent > 30 else 25,
            "learning_style": "adaptive"
        }

    async def schedule_spaced_repetition(self, memory_ids: List[str]) -> Dict[str, Any]:
        """Schedule spaced repetition using SM-2 algorithm.
        
        SM-2 Algorithm:
        - EF (Easiness Factor): starts at 2.5, adjusted based on quality
        - Quality: 0-5 scale (0=complete blackout, 5=perfect response)
        - Interval calculation based on EF and repetition count
        
        Args:
            memory_ids: List of memory/item IDs to schedule
            
        Returns:
            Spaced repetition schedule with intervals
        """
        schedule = {}
        for memory_id in memory_ids:
            if memory_id not in self.spaced_repetition_schedule:
                self.spaced_repetition_schedule[memory_id] = {
                    "easiness_factor": 2.5,
                    "interval": 1,
                    "repetitions": 0,
                    "next_review": None,
                    "quality_history": []
                }
            
            entry = self.spaced_repetition_schedule[memory_id]
            schedule[memory_id] = {
                "interval_days": entry["interval"],
                "easiness_factor": entry["easiness_factor"],
                "repetitions": entry["repetitions"],
                "next_review_days": entry["interval"]
            }
        
        return {"scheduled": True, "count": len(memory_ids), "schedule": schedule}

    async def update_memory_strength(self, memory_id: str, quality: int) -> Dict[str, Any]:
        """Update memory strength using SM-2 algorithm.
        
        Args:
            memory_id: Memory identifier
            quality: Response quality (0-5)
            
        Returns:
            Updated memory parameters
        """
        quality = max(0, min(5, quality))
        
        if memory_id not in self.spaced_repetition_schedule:
            self.spaced_repetition_schedule[memory_id] = {
                "easiness_factor": 2.5,
                "interval": 1,
                "repetitions": 0,
                "quality_history": []
            }
        
        entry = self.spaced_repetition_schedule[memory_id]
        entry["quality_history"].append(quality)
        
        if quality < 3:
            entry["repetitions"] = 0
            entry["interval"] = 1
        else:
            if entry["repetitions"] == 0:
                entry["interval"] = 1
            elif entry["repetitions"] == 1:
                entry["interval"] = 6
            else:
                entry["interval"] = round(entry["interval"] * entry["easiness_factor"])
            entry["repetitions"] += 1
        
        entry["easiness_factor"] = max(1.3, entry["easiness_factor"] + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        
        return {
            "memory_id": memory_id,
            "new_interval": entry["interval"],
            "new_easiness_factor": entry["easiness_factor"],
            "repetitions": entry["repetitions"],
            "memorized": quality >= 3
        }

    async def consolidate_knowledge(self, topic: str, related_items: List[str]) -> Dict[str, Any]:
        """Consolidate related knowledge items into coherent structure.
        
        Args:
            topic: Main topic name
            related_items: List of related memory/content IDs
            
        Returns:
            Consolidated knowledge summary
        """
        if not related_items:
            return {"topic": topic, "consolidated": False, "items": []}
        
        item_data = []
        for item_id in related_items:
            if item_id in self.spaced_repetition_schedule:
                entry = self.spaced_repetition_schedule[item_id]
                item_data.append({
                    "id": item_id,
                    "strength": entry["easiness_factor"] * entry["repetitions"],
                    "interval": entry["interval"]
                })
        
        item_data.sort(key=lambda x: x["strength"], reverse=True)
        
        core_items = [item["id"] for item in item_data[:5]]
        supporting_items = [item["id"] for item in item_data[5:]]
        
        return {
            "topic": topic,
            "consolidated": True,
            "core_knowledge": core_items,
            "supporting_knowledge": supporting_items,
            "total_items": len(related_items),
            "strength_score": sum(item["strength"] for item in item_data) / len(item_data) if item_data else 0
        }
