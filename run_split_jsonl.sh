#!/bin/bash

datadir=$1

for path in $datadir/*; do
    relname=$(basename "$path")
    filepath="$path/$relname.jsonl"
    python -m dataset.split_jsonl $filepath $path --train_ratio 0.8 --val_ratio 0.2
    echo "Split $relname"
done
