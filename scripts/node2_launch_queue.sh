#!/usr/bin/env bash
# Detached launcher for the node2 queue worker.
cd /home/user/ultratensor-cluster/scripts || exit 1
mkdir -p logs
: > logs/node2_queue.jsonl
setsid bash node2_queue.sh node2_prompts.txt logs/node2_queue.jsonl 32 \
  > logs/node2_queue.log 2>&1 < /dev/null &
echo "launched pid $!"
sleep 3
ls -la logs/
tail -5 logs/node2_queue.log 2>/dev/null
