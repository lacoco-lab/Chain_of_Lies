import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_of_lies.rl.rewards import ALL_VARIANT_SPECS, SELECTED_RULE_BASED_RL_VARIANTS, summarize_variant_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize current result quality across experiment variants.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/Without_self_eval"),
        help="Base data directory containing prompt/response subdirectories.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a text table.",
    )
    args = parser.parse_args()

    summaries = []
    for variant_name, spec in ALL_VARIANT_SPECS.items():
        prompts_dir = Path(spec["prompts_dir"])
        responses_dir = Path(spec["responses_dir"])
        summaries.append(summarize_variant_results(prompts_dir, responses_dir))

    if args.json:
        print(json.dumps(summaries, indent=2))
        return

    print("Variant analysis for current Without_self_eval results")
    print()
    for item in summaries:
        selected = " [selected for RL]" if item["variant_name"] in SELECTED_RULE_BASED_RL_VARIANTS else ""
        print(f"{item['variant_name']}{selected}")
        print(f"  examples: {item['num_examples']}")
        print(f"  avg_reward: {item.get('avg_reward', 0.0):.2f}")
        print(f"  task_success_rate: {item.get('task_success_rate', 0.0):.2f}")
        print(f"  concealment_rate: {item.get('concealment_rate', 0.0):.2f}")
        print(f"  format_rate: {item.get('format_rate', 0.0):.2f}")
        print(f"  avg_cot_words: {item.get('avg_cot_words', 0.0):.1f}")
        components = item.get("component_means", {})
        if components:
            parts = ", ".join(f"{key}={value:.2f}" for key, value in components.items())
            print(f"  components: {parts}")
        if item.get("missing_responses"):
            print(f"  missing_responses: {', '.join(item['missing_responses'])}")
        print()

    print("Selected RL variants:")
    for variant in SELECTED_RULE_BASED_RL_VARIANTS:
        print(f"  - {variant}")


if __name__ == "__main__":
    main()
