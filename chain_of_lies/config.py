"""
Default configuration for graph generation and pipeline.

Override via environment variables or by passing explicit args to each stage.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GraphConfig:
    """Parameters for synthetic graph generation."""

    n_nodes: int = 20
    edge_probability: float = 0.25  # or use n_edges for deterministic
    seed: Optional[int] = None
    # Ensure graph is connected (e.g. add edges until connected)
    ensure_connected: bool = True


@dataclass
class PipelinePaths:
    """Where to read/write data between stages."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    prompts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "prompts")
    responses_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "responses")
    results_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "results")

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.prompts_dir, self.responses_dir, self.results_dir):
            p.mkdir(parents=True, exist_ok=True)


def get_default_paths() -> PipelinePaths:
    return PipelinePaths()


def get_default_graph_config() -> GraphConfig:
    return GraphConfig()
