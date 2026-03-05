# Chain-of-Lies

AI safety research: encoded reasoning (steganography in Chain-of-Thought). Pipeline: **Stage 1** (graph + prompt generation) and **Stage 2** (Qwen local inference).

## Setup

```bash
pip install -r requirements.txt
```

## Stage 1: Generate prompts

```bash
python scripts/run_stage1.py --n 10 --seed 42
# Writes data/prompts/*.json
```

## Stage 2: Run inference (local)

Uses **Qwen 2.5 or 3** via Hugging Face (no API keys):

```bash
python scripts/run_inference.py --model Qwen/Qwen2.5-7B-Instruct
python scripts/run_inference.py --model Qwen/Qwen2.5-3B-Instruct --resume
```

Responses are written to `data/responses/<experiment_id>.json`. Use `--resume` to skip experiments that already have a response.

## Running on HTCondor

Inference can run on your cluster so it keeps running when you close your laptop. Progress is visible with `tail -f /scratch/$USER/logs/chain_of_lies/job.out`.

See **[condor/README.md](condor/README.md)** for:

- One-time setup (log dir, rsync, venv, `job.sub` paths)
- Submit and monitor (`condor_submit`, `condor_q`, `tail -f` job.out)
- Resume after interrupt (`--resume` in `condor/run.sh`)

## Project layout

```
chain_of_lies/
  stage1_graph_prompt/   # Graph + prompt generation
  stage2_inference/      # Qwen local inference
condor/                  # HTCondor run.sh, job.sub, README
scripts/
  run_stage1.py          # Generate prompts
  run_inference.py       # Batch inference (progress + resume)
data/
  prompts/               # Input: prompt JSONs
  responses/             # Output: response JSONs + progress.txt
```
