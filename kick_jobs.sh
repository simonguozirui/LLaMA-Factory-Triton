#!/bin/bash

# Define the list of configuration files
CONFIG_FILES=(
    "night_configs/qwen_1.yaml"
    "night_configs/qwen_2.yaml"
    "night_configs/qwen_3.yaml"
    "night_configs/qwen_4.yaml"
    "night_configs/qwen_5.yaml"
    "night_configs/qwen_6.yaml"
    "night_configs/qwen_7.yaml"
)

# Iterate through each configuration file and run training
for config in "${CONFIG_FILES[@]}"; do
    echo "Running training with $config..."
    llamafactory-cli train "$config"
done

echo "All training runs completed!"