#!/bin/bash

prompt_format_filename="misc/prompt_formats.txt"
manual_prompts_filename="misc/manual_prompts.txt"

datadir1=$1
logdir1=$2
datadir2=$3
logdir2=$4
# datadir3=$5
# logdir3=$4
# datadir4=$7
# logdir4=$5

mkdir -p $logdir1
mkdir -p $logdir2
# mkdir -p $logdir3
# mkdir -p $logdir4

# Unconditional probing -> BERT bsz = 64, RoBERTa bsz = 48
# Conditional probing -> BERT bsz = 32, RoBERTa bsz = 8

# UN CONDITIONAL
i=1
for path in $datadir1/*; do
    filename=$(basename "$path")
    logfile="$logdir1/$filename.txt"
    prompt_format=$(sed -n ${i}p $prompt_format_filename)
    manual_prompt=$(sed -n ${i}p $manual_prompts_filename)
    # if [ "$filename" != "P1001" ] && [ "$filename" != "P101" ] && [ "$filename" != "P103" ] && [ "$filename" != "P106" ] && [ "$filename" != "P108" ] && [ "$filename" != "P127" ] && [ "$filename" != "P1303" ] && [ "$filename" != "P131" ] && [ "$filename" != "P136" ] && [ "$filename" != "P1376" ] && [ "$filename" != "P138" ] && [ "$filename" != "P140" ] && [ "$filename" != "P1412" ] && [ "$filename" != "P159" ] && [ "$filename" != "P17" ]; then
    echo "Creating trigger for $filename"
    $(CUDA_VISIBLE_DEVICES=1 python -m main.create_trigger $path out --model_name "bert-base-cased" --iters 50 --bsz 64 --patience 10 --num_cand 10 --format "X-7-Y" > $logfile)
    echo "Saving results to $logfile"
    # fi
    ((i++))
done
echo "--------------------------------------------------------------"

# UNCONDITIONAL
i=1
for path in $datadir2/*; do
    filename=$(basename "$path")
    logfile="$logdir2/$filename.txt"
    prompt_format=$(sed -n ${i}p $prompt_format_filename)
    manual_prompt=$(sed -n ${i}p $manual_prompts_filename)
    echo "Creating trigger for $filename"
    $(CUDA_VISIBLE_DEVICES=1 python -m main.create_trigger $path out --model_name "roberta-large" --iters 50 --bsz 64 --patience 10 --num_cand 10 --format "X-5-Y" > $logfile)
    echo "Saving results to $logfile"
    ((i++))
done
echo "--------------------------------------------------------------"

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
