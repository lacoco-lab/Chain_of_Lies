# Benchmark data notes

These are small, manually curated, representative examples—not copies of the filler-token
paper's complete evaluation datasets.

- `one_fact` mirrors its retrieve-one-fact-then-add structure using ages at death.
- `two_fact` mirrors retrieval of two atomic numbers followed by addition.
- `two_hop` mirrors capital/element lookup followed by letter extraction.
- `system_equations` uses prompt-local nonce variables and a computed reference chain.

The evaluation facts are disjoint from the few-shot demonstrations. Every factual main item
references one or two records in `component_facts.jsonl`, allowing the report to distinguish
failed component probes from failed composition.

Gold arithmetic, component references, bridge strings, and requested letter positions are checked
by `python3 preliminary_filler_gate/run.py validate`. The validator cannot establish that every
natural-language fact is uncontroversial; revisions should therefore be reviewed before treating
this small screen as a frozen benchmark.

To add an item:

1. give it a unique deterministic ID;
2. use one of the four registered task types;
3. add all factual prerequisites to `component_facts.jsonl`;
4. reference those component IDs from the main item;
5. provide machine-checkable metadata in the same shape as neighboring records;
6. rerun validation and unit tests.

