#!/bin/sh
# Run the compute-reduced CPU experiment queue (resumable: finished seeds are skipped, unfinished ones resume).
cd "$(dirname "$0")/.." || exit 1
exec python3 scripts/run_experiment.py \
  configs/cpu/C_ST_risk_head128.json configs/cpu/C_ST_risk_headtail256.json \
  configs/cpu/C_ST_sentiment_muril.json configs/cpu/C_ST_sentiment_hingroberta.json \
  configs/cpu/C_ST_emotion_muril.json configs/cpu/C_MT_muril.json configs/cpu/C_ST_risk_head256.json --device cpu
