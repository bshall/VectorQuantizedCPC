#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#



set -e

NUM_CORES=6
export PYTHONUNBUFFERED="YOUR_SET"
checkpoint_dir="checkpoints/ZeroSpeech2019/english"
in_dir="datasets/ZeroSpeech2019/english/test"
out_dir="encoded/ZeroSpeech2019/english/test"
abx_out_dir="encoded/ZeroSpeech2019/english/test"
task_path="data/byCtxt_acSpkr.abx"
result_file=abx_scores.txt
abx_dir="abx/"
encoded_dir="encoded/"

if [ -f $result_file ]; then
	rm -r $result_file
fi

for entry in "$checkpoint_dir"/*
do
	if [ -d $encoded_dir ]; then
		echo "Removing encoded folder"
		rm -r $encoded_dir
	fi
	python encode.py --checkpoint="$entry" --in-dir="$in_dir" --out-dir="$out_dir"
	if [ -d $abx_dir ]; then
		echo "Removing abx folder"
		rm -r $abx_dir
	fi
	result=$(python abx.py --task-type="across" --task-path="$task_path" --feature-dir="$out_dir" --out-dir="$abx_out_dir" | grep "average") 
	echo "$entry" $result >> $result_file

done
