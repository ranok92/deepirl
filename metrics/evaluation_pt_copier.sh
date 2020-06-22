#!/bin/bash

ALL_RES_PATH=$1
echo "Searching for pytorch .pt files in $ALL_RES_PATH ..."

LOW_SVF_FINDER_DIR=$(pwd)/low_svf_finder.py
cd $ALL_RES_PATH

for res_type in ./*; do
    echo "found result type" $res_type
    for seed in $res_type/*; do
        # low svf symlink
        if [ -d "$seed"/tf_logs/ ]; then
            low_svf="$(python -W ignore $LOW_SVF_FINDER_DIR --tf-file $seed/tf_logs/* | sed -nr 's/.*step\s([0-9]+)/\1/p').pt"
            echo "creating symlink $seed/$low_svf into $seed"
            ln -sf ./saved-models/$low_svf "$seed"/"low_svf_$low_svf"
        fi

        # latest iteration symlink
        if [ -d "$seed"/saved-models/ ]; then
            last_iter="$(ls -v $seed/saved-models | tail -1)"
            echo "creating symlink $seed/$last_iter into $seed"
            ln -sf ./saved-models/$last_iter "$seed"/"last_iter_$last_iter"
        fi
        if [ -d "$seed"/policy-models/ ]; then
            last_iter="$(ls -v "$seed"/policy-models | tail -1)"
            echo "creating symlink $seed/$last_iter into $seed"
            ln -sf ./policy-models/$last_iter "$seed"/"last_iter_$last_iter"
        fi
    done
done
