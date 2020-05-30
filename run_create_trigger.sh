#!/bin/bash

prompt_format_filename="misc/prompt_formats.txt"
manual_prompts_filename="misc/manual_prompts.txt"

datadir1=$1
logdir1=$2
# datadir2=$3
# logdir2=$4
# datadir3=$5
# logdir3=$4
# datadir4=$7
# logdir4=$5

mkdir -p $logdir1
# mkdir -p $logdir2
# mkdir -p $logdir3
# mkdir -p $logdir4

# Unconditional probing -> batch size = 64
# Conditional probing -> batch size = 32

# UNCONDITIONAL
i=1
for path in $datadir1/*; do
    filename=$(basename "$path")
    logfile="$logdir1/$filename.txt"
    prompt_format=$(sed -n ${i}p $prompt_format_filename)
    manual_prompt=$(sed -n ${i}p $manual_prompts_filename)
    echo "Creating trigger for $filename"
    python -m main.create_trigger $path out --model_name "bert-base-cased" --iters 50 --bsz 64 --patience 10 --num_cand 10 --format "2-X-2-Y-2" > $logfile
    echo "Saving results to $logfile"
    ((i++))
done
echo "--------------------------------------------------------------"

# # CONDITIONAL
# i=1
# for path in $datadir2/*; do
#     filename=$(basename "$path")
#     logfile="$logdir2/$filename.txt"
#     prompt_format=$(sed -n ${i}p $prompt_format_filename)
#     manual_prompt=$(sed -n ${i}p $manual_prompts_filename)
#     echo "Creating trigger for $filename"
#     python -m main.create_trigger $path out --model_name "roberta-large" --iters 50 --bsz 8 --patience 10 --num_cand 50 --format "X-5-Y" --use_ctx > $logfile
#     echo "Saving results to $logfile"
#     ((i++))
# done
# echo "--------------------------------------------------------------"

# # UNCONDITIONAL
# i=1
# for path in $datadir/*; do
#     filename=$(basename "$path")
#     logfile="$logdir3/$filename.txt"
#     prompt_format=$(sed -n ${i}p $prompt_format_filename)
#     manual_prompt=$(sed -n ${i}p $manual_prompts_filename)
#     echo "Creating trigger for $filename"
#     python -m main.create_trigger $path out --model_name "bert-base-cased" --iters 50 --bsz 64 --patience 10 --num_cand 10 --format "X-7-Y" > $logfile
#     echo "Saving results to $logfile"
#     ((i++))
# done
# echo "--------------------------------------------------------------"

# # UNCONDITIONAL
# i=1
# for path in $datadir/*; do
#     filename=$(basename "$path")
#     logfile="$logdir4/$filename.txt"
#     prompt_format=$(sed -n ${i}p $prompt_format_filename)
#     manual_prompt=$(sed -n ${i}p $manual_prompts_filename)
#     echo "Creating trigger for $filename"
#     python -m main.create_trigger $path out --model_name "bert-base-cased" --iters 50 --bsz 64 --patience 10 --num_cand 50 --format "X-7-Y" > $logfile
#     echo "Saving results to $logfile"
#     ((i++))
# done
# echo "--------------------------------------------------------------"
