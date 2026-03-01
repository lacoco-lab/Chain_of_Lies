"""
Chain-of-Lies: AI Safety research on encoded reasoning (steganography in Chain-of-Thought).

Pipeline stages:
  1. graph_prompt  — Generate synthetic graphs and LLM prompts
  2. inference     — Run LLM inference and record outputs
  3. parsing       — Extract CoT trace and <ANSWER> paths
  4. evaluation    — Validate paths, trace length, intervention sensitivity
"""

__version__ = "0.1.0"
