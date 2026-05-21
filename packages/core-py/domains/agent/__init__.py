"""
Agent Domain — agentic infrastructure for SloNet/SloEngine.

An *agent* wraps a model with tools, memory, and a reasoning loop:
  perceive → think → act → observe → repeat

Tools are Python functions registered with a name + description.
The model decides which tool to call based on structured text prompts.
"""

from .engine import AgentEngine, Tool, AgentRun
