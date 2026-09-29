"""CPU PRE gate: wait for preserved old jobs and their transferred checkpoints."""

import argparse
from compact import query, identity
from common import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", type=int, action="append", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    try:
        rows = [row for controller in a.controller for row in query(controller)]
    except Exception as error:
        print("Waiting: could not query old jobs:", error)
        return 75
    for row in rows:
        ident = identity(row)
        if (
            ident
            and ident[1:] == (a.task, a.model, a.seed)
            and row["JobStatus"] in [1, 2, 4, 5, 6, 7]
        ):
            print(
                f'Waiting for old {ident[0]} job {row["ClusterId"]}.{row["ProcId"]}; status={row["JobStatus"]}'
            )
            return 75
    try:
        checkpoint_valid(
            condition(a.task, a.model, a.seed),
            a.task,
            a.model,
            a.seed,
            load_config(CONFIG),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Final checkpoint not valid; refusing GPU evaluation:", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
