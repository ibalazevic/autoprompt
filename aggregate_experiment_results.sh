#!/bin/bash

datadir=$1
outfile=$2

for path in $datadir/*; do
    filename=$(basename "$path")
    echo $filename >> $outfile
    tail -3 $path >> $outfile
done
