"""
Stage 2: LLM Inference.

Connect to LLM API, send generated prompts, and record raw responses.
"""

from chain_of_lies.stage2_inference.api import run_inference

__all__ = ["run_inference"]
