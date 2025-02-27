#!/bin/bash

# 设置chunk-size参数
chunk_sizes=(128)

# 设置data-folder参数
data_folders=(
    "/data/scifact"
    "/data/nfcorpus"
    "/data/treccovid"
)

# 设置pooling-alg参数
pooling_algs=(late-chunking naive-chunking hie-chunking)

# 输出文件
output_file="results.txt"

# 清空输出文件
echo "" > "$output_file"

# 三重循环
for chunk_size in "${chunk_sizes[@]}"; do
    for data_folder in "${data_folders[@]}"; do
        for pooling_alg in "${pooling_algs[@]}"; do
            echo "Running with chunk-size=$chunk_size, data-folder='$data_folder', pooling-alg='$pooling_alg'" | tee -a "$output_file"
            python run_chunked_eval.py \
                --model-name '/share/yangxinhao-local/hf_models/jina-embeddings-v2-small-en' \
                --chunk-size "$chunk_size" \
                --data-folder "$data_folder" \
                --task-name SciFactChunked \
                --pooling-alg "$pooling_alg" \
                >> "$output_file" 2>&1
        done
    done
done