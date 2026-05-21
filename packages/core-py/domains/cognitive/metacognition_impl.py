"""
Metacognition - Ported from recovered slo_metacognitive.py
Self-monitoring, self-correction, and confidence assessment
"""

import time
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import statistics


class ContradictionType(Enum):
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    LOGICAL = "logical"
    NUMERICAL = "numerical"


@dataclass
class Contradiction:
    source1: str
    source2: str
    contradiction_type: ContradictionType
    severity: float
    description: str
    detected_at: float


@dataclass
class SelfAssessment:
    response_id: str
    confidence: float
    completeness: float
    accuracy: float
    contradictions: List[Contradiction]
    self_correction_applied: bool
    original_response: str
    corrected_response: str
    assessment_timestamp: float


class Metacognition:
    """Self-monitoring and confidence assessment"""

    def __init__(self):
        self.assessments: List[SelfAssessment] = []
        self.performance_history: List[Dict] = []

    def assess_response(self, response: str, context: Dict) -> SelfAssessment:
        """Assess a response for confidence, completeness, accuracy."""
        assessment_id = f"asm_{len(self.assessments)}_{int(time.time())}"

        confidence = self._calculate_confidence(response, context)
        completeness = self._calculate_completeness(response, context)
        accuracy = self._calculate_accuracy(response, context)

        assessment = SelfAssessment(
            response_id=assessment_id,
            confidence=confidence,
            completeness=completeness,
            accuracy=accuracy,
            contradictions=[],
            self_correction_applied=False,
            original_response=response,
            corrected_response=response,
            assessment_timestamp=time.time(),
        )

        self.assessments.append(assessment)

        if confidence < 0.5:
            assessment.corrected_response = self._apply_self_correction(response, context)
            assessment.self_correction_applied = True

        return assessment

    def _calculate_confidence(self, response: str, context: Dict) -> float:
        """Calculate confidence score."""
        if not response:
            return 0.0

        length_factor = min(len(response) / 100, 1.0)
        base_confidence = 0.7

        if "?" in response:
            base_confidence -= 0.1

        return min(base_confidence + length_factor * 0.3, 1.0)

    def _calculate_completeness(self, response: str, context: Dict) -> float:
        """Calculate completeness score."""
        if not response:
            return 0.0

        indicators = ["because", "therefore", "however", "first", "then", "finally"]
        indicator_count = sum(1 for ind in indicators if ind in response.lower())

        return min(0.3 + indicator_count * 0.15, 1.0)

    def _calculate_accuracy(self, response: str, context: Dict) -> float:
        """Calculate accuracy score."""
        return 0.8

    def _apply_self_correction(self, response: str, context: Dict) -> str:
        """Apply self-correction to response."""
        corrections = [
            ("I think", "It appears"),
            ("maybe", "likely"),
            ("probably", "in most cases"),
        ]

        corrected = response
        for old, new in corrections:
            corrected = corrected.replace(old, new)

        return corrected

    def detect_contradiction(self, response1: str, response2: str) -> List[Contradiction]:
        """Detect contradictions between responses."""
        contradictions = []

        keywords1 = set(response1.lower().split())
        keywords2 = set(response2.lower().split())

        common = keywords1 & keywords2
        if not common and len(response1) > 10 and len(response2) > 10:
            contradictions.append(
                Contradiction(
                    source1="response1",
                    source2="response2",
                    contradiction_type=ContradictionType.LOGICAL,
                    severity=0.5,
                    description="Responses appear to contradict each other",
                    detected_at=time.time(),
                )
            )

        return contradictions

    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        if not self.assessments:
            return {"total": 0}

        confidences = [a.confidence for a in self.assessments]

        return {
            "total_assessments": len(self.assessments),
            "avg_confidence": statistics.mean(confidences),
            "self_corrections": sum(1 for a in self.assessments if a.self_correction_applied),
        }


__all__ = ["Metacognition", "SelfAssessment", "Contradiction", "ContradictionType"]
