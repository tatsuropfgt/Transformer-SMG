#!/bin/bash

# 設定ファイルのリスト
musicrep=("note_tuple" "piano_roll" "remi_plus")
config_settings=("baseline" "rel_idx_a01" "rel_idx_time_pitch_linear_had_a01" "rel_idx_time_pitch_linear_a01_baseline_sin" "rel_idx_time_pitch_linear_a01")

# 各設定ファイルで python generate.py を実行
for mr in "${musicrep[@]}"
do
    for cs in "${config_settings[@]}"
    do
        config_file="configs/${mr}/${cs}.yaml"
        echo "Running python generate.py --config $config_file --given_bar 4 --gen_num 15"
        python generate.py --config $config_file --given_bar 4 --gen_num 15
    done
done
