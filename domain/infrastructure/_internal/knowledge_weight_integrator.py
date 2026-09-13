"""Backward-compatibility shim."""
import domains.infrastructure.knowledge_weight_integrator as _mod
globals().update(vars(_mod))
