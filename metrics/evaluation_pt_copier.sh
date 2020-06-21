#!/bin/bash

ALL_RES_PATH=$1
echo "Searching for pytorch .pt files in $ALL_RES_PATH ..."

LOW_SVF_FINDER_DIR=low_svf_finder.py

for res_type in $ALL_RES_PATH/*; do
    echo "found result type" $res_type
    for seed in $res_type/*; do
       # low svf symlink
       low_svf="$(python -W ignore $LOW_SVF_FINDER_DIR --tf-file $seed/tf_logs/* | sed -nr 's/.*step\s([0-9]+)/\1/p').pt"
       echo "creating symlink $seed/$low_svf into $seed"
       ln -s $seed/$low_svf $seed

       # latest iteration symlink
       last_iter="$(ls -v $seed/saved-models | tail -1)"
       echo "creating symlink $seed/$last_iter into $seed"
       ln -s $seed/$last_iter $seed
    done
done