#!/bin/bash

# Checkpoints for FashionMNIST
CKPT_FLOW_FASHION="jac-zac/bayesflow-project/best-model:v127"
CKPT_DIFF_FASHION="jac-zac/bayesflow-project/best-model:v145"

METHODS=("diffusion" "flow")

for method in "${METHODS[@]}"; do

  if [ "$method" == "flow" ]; then
    CKPT=$CKPT_FLOW_FASHION
    STEPS=15
    SLICE_START=1
    SLICE_END=2
  else
    CKPT=$CKPT_DIFF_FASHION
    STEPS=50
    SLICE_START=1
    SLICE_END=2
  fi

  echo "Running FashionMNIST with $method, steps=$STEPS, slice=[$SLICE_START:$SLICE_END]"

  python -m src.eval.activation_study \
    --ckpt "$CKPT" \
    --dataset-name "FashionMNIST" \
    --method "$method" \
    --batch-size 16 \
    --steps "$STEPS" \
    --cov-samples 100 \
    --slice-start "$SLICE_START" \
    --slice-end "$SLICE_END"

done
