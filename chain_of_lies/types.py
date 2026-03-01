"""
Shared data types for the Chain-of-Lies pipeline.

These types are used across stages so that graph generation, prompts,
LLM outputs, and evaluation all share a consistent structure.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GraphSpec:
    """Specification of a generated graph (nodes, edges, partition, targets)."""

    edges_text: str  # Text list of edges for the prompt, e.g. "Node_1 <-> Node_2"
    start_node: str
    public_target: str
    private_target: str
    node_list: list[str] = field(default_factory=list)
    edge_list: list[tuple[str, str]] = field(default_factory=list)
    # Optional: for evaluation and intervention tests
    public_distance: Optional[int] = None
    private_distance: Optional[int] = None


@dataclass
class ExperimentPrompt:
    """Full experiment: graph spec + the exact prompt string for the LLM."""

    prompt_text: str
    spec: GraphSpec
    experiment_id: str = ""


@dataclass
class LLMResponse:
    """Raw response from the LLM (full completion)."""

    raw_text: str
    experiment_id: str = ""
    model_id: str = ""


@dataclass
class ParsedOutput:
    """Parsed response: CoT trace and answer block."""

    chain_of_thought: str
    answer_block: str
    experiment_id: str = ""
    # Extracted paths (if parsed from <ANSWER>...</ANSWER>)
    public_path: Optional[list[str]] = None
    private_path: Optional[list[str]] = None
