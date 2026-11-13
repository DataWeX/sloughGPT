"""Rules package — condition → action engine plus dict DSL."""

from avion.rules.dsl import rules_from_dict
from avion.rules.engine import Rule, RuleAction, RuleEngine

__all__ = ["Rule", "RuleAction", "RuleEngine", "rules_from_dict"]
