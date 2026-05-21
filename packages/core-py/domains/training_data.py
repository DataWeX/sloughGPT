"""
Training Data Collector

Collects conversation data for training the AI companion.
Stores user/assistant pairs for later fine-tuning.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ConversationPair:
    """Single conversation turn."""
    user: str
    assistant: str
    timestamp: str
    model: str
    rating: Optional[int] = None  # thumbs up/down


class TrainingDataCollector:
    """
    Collect training data from conversations.
    
    Usage:
        collector = TrainingDataCollector()
        
        # Log a conversation
        collector.log(
            user_message="Hey!",
            assistant_response="Hey! How's it going?",
            model="gpt2",
        )
        
        # Get collected data
        pairs = collector.get_pairs()
        
        # Export for training
        collector.export_for_training("data/conversations.jsonl")
    """
    
    def __init__(self, log_dir: str = "data/training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = self.log_dir / f"conversations_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    def log(
        self,
        user_message: str,
        assistant_response: str,
        model: str,
        rating: Optional[int] = None,
    ) -> ConversationPair:
        """Log a conversation pair."""
        pair = ConversationPair(
            user=user_message,
            assistant=assistant_response,
            timestamp=datetime.now().isoformat(),
            model=model,
            rating=rating,
        )
        
        # Append to file
        with open(self.current_file, "a") as f:
            f.write(json.dumps(asdict(pair)) + "\n")
        
        return pair
    
    def get_pairs(
        self,
        limit: int = 1000,
        min_length: int = 5,
    ) -> List[ConversationPair]:
        """Get conversation pairs."""
        pairs = []
        
        for log_file in sorted(self.log_dir.glob("*.jsonl")):
            with open(log_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        
                        # Filter short responses
                        if len(data.get("assistant", "")) < min_length:
                            continue
                        
                        pairs.append(ConversationPair(**data))
                    except:
                        continue
        
        return pairs[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        pairs = self.get_pairs(limit=10000)
        
        if not pairs:
            return {"total": 0}
        
        avg_length = sum(len(p.assistant.split()) for p in pairs) / len(pairs)
        ratings = [p.rating for p in pairs if p.rating is not None]
        
        return {
            "total_pairs": len(pairs),
            "avg_response_length": round(avg_length, 1),
            "rated_count": len(ratings),
            "positive_rate": sum(ratings) / len(ratings) if ratings else 0,
            "log_file": str(self.current_file),
        }
    
    def export_for_training(
        self,
        output_path: str,
        format: str = "jsonl",  # jsonl, alpaca, or chatml
    ) -> str:
        """Export in training format."""
        pairs = self.get_pairs(limit=10000)
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, "w") as f:
            for pair in pairs:
                if format == "alpaca":
                    data = {
                        "instruction": pair.user,
                        "output": pair.assistant,
                        "input": "",
                    }
                elif format == "chatml":
                    data = {
                        "messages": [
                            {"role": "user", "content": pair.user},
                            {"role": "assistant", "content": pair.assistant},
                        ]
                    }
                else:  # jsonl
                    data = {
                        "user": pair.user,
                        "assistant": pair.assistant,
                    }
                
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        return str(output)
    
    def get_high_quality_pairs(
        self,
        min_rating: int = 1,  # thumbs up = 1
    ) -> List[ConversationPair]:
        """Get pairs with positive feedback."""
        all_pairs = self.get_pairs(limit=10000)
        return [p for p in all_pairs if p.rating and p.rating >= min_rating]


# Global instance
_collector: TrainingDataCollector = None


def get_collector() -> TrainingDataCollector:
    """Get training data collector."""
    global _collector
    if _collector is None:
        _collector = TrainingDataCollector()
    return _collector


__all__ = [
    "ConversationPair",
    "TrainingDataCollector",
    "get_collector",
]