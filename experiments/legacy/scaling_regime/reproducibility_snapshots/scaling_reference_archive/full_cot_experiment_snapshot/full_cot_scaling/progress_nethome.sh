#!/bin/bash
# Two separate invocations estimate evaluation rate; no Python or blocking waits.
set -euo pipefail
task_repo=${FULL_COT_PROGRESS_ROOT:-/nethome/mmohammadkhani/Faithfulness-Safety}
task_logs=${FULL_COT_PROGRESS_LOGS:-/scratch/mmohammadkhani/logs/chain_of_lies}
task_state=${FULL_COT_PROGRESS_STATE:-/scratch/mmohammadkhani/full_cot_progress}
mkdir -p "$task_state"
task_now=$(date +%s)
condor_q mmohammadkhani -constraint 'ClusterId == 119862 && JobStatus == 2' \
  -af ClusterId ProcId JobCurrentStartDate Args |
while read -r task_cluster task_proc task_start task_runner task_name task_model task_seed task_action; do
  case "$task_name" in
    s5) task_total=3800 ;;
    knowledge|parity) task_total=1000 ;;
    multiplication) task_total=600 ;;
    *) continue ;;
  esac
  task_responses="$task_repo/generated_data/full_cot_scaling_v1_eval_responses/$task_name/$task_model/seed_$task_seed/all"
  task_count=0
  if [[ -d "$task_responses" ]]; then
    task_count=$(find "$task_responses" -maxdepth 1 -type f -name '*.json' | wc -l)
  fi
  task_snapshot="$task_state/$task_cluster.$task_proc.snapshot"
  printf 'JOB %s.%s | %s %s seed %s | saved files %s/%s\n' \
    "$task_cluster" "$task_proc" "$task_name" "$task_model" "$task_seed" "$task_count" "$task_total"
  if [[ -f "$task_snapshot" ]]; then
    read -r task_previous_time task_previous_count task_previous_start < "$task_snapshot"
    if [[ "$task_start" == "$task_previous_start" ]]; then
      awk -v now="$task_now" -v before="$task_previous_time" -v count="$task_count" \
        -v old="$task_previous_count" -v total="$task_total" 'BEGIN {
          elapsed=now-before; delta=count-old;
          if (elapsed>0 && delta>0) {
            rate=delta/elapsed;
            printf "  Recent rate: %.1f prompts/min; estimated evaluation remaining: %.1f hours\n", rate*60, (total-count)/rate/3600;
          } else print "  No new saved responses since last check; inspect logs/GPU (not proof of a stall).";
        }'
    else
      echo '  Worker start changed; rate baseline reset.'
    fi
  else
    echo '  First observation. Run again in 10–15 minutes for a rate-based estimate.'
  fi
  printf '%s %s %s\n' "$task_now" "$task_count" "$task_start" > "$task_snapshot.tmp"
  mv "$task_snapshot.tmp" "$task_snapshot"
  task_log="$task_logs/full_cot_${task_name}_${task_model}_seed_${task_seed}.out"
  if [[ -f "$task_log" ]]; then
    tail -n 150 "$task_log" | awk '/\[Eval\]|\[CE\]|\[Full-CoT\]|\[Stage/ {line=$0} END {if(line!="") print "  Latest log: " line}'
  fi
done
