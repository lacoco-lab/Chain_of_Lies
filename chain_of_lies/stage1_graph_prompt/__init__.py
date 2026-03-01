"""
Stage 1: Graph & Prompt Generation.

Generate synthetic connected graphs (via NetworkX),
partition nodes into Public/Private, select start and targets, and produce
the exact prompt string for the LLM.
"""

from chain_of_lies.stage1_graph_prompt.generate import generate_experiment_prompt

__all__ = ["generate_experiment_prompt"]
