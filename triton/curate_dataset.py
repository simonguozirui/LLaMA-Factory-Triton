"""
Convert a dataset file into 
dataset format for Llama Factory (Alpaca)
[
  {
    "instruction": "human instruction (required)",
    "input": "human input (optional)",
    "output": "model response (required)",
  }
]
"""
import json
import argparse

import os
from tqdm import tqdm
from datasets import load_dataset

import toml
import string


# TRITON_INSTRUCTION = "Convert this PyTorch neural network code to its optimized Triton implementation."

def construct_instruction(entrypoint_name):
    # Load the TOML configuration
    config = toml.load("triton/triton_prompt_no_ex_template.toml")

    # Format the prompt with the architecture name
    formatted_prompt = config["prompt"].format(entrypoint_name=entrypoint_name)

    return formatted_prompt

def convert_dataset_from_huggingface(dataset_name, output_file, limit:int=None):
    # Load the dataset from HuggingFace

    dataset = load_dataset(dataset_name)
    
    # # Most HF datasets have a 'train' split
    if 'train' in dataset:
        data = dataset['train']
    else:
        # Use the first available split if no 'train' split
        data = dataset[list(dataset.keys())[0]]

    # limit
    if limit:
        limit = min(limit, data.num_rows)
        data = data.select(range(limit))

    # Convert to Alpaca format
    # Apply limit if specified (for debugging)
    print(f"Dataset {dataset_name} contains {data.num_rows} rows, processing with limit {limit}")

    # final converted data
    converted_data = []

    for item in tqdm(data):
        entry_point = item["entry_point"]
        uuid = item["uuid"]
        converted_item = {
            "instruction": construct_instruction(entry_point),
            "input": item["python_code"],
            "output": item["triton_code"],
            "entry_point": entry_point,
            "uuid": uuid
        }

        converted_data.append(converted_item)
    
    # # Write to output file
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)

if __name__ == "__main__":


    # # GitHub Scrape
    out_dataset_dir = "/matx/u/simonguo/triton_sft_data"
    limit = 18500

    out_file_name = f"pytorch_scrape_github_inductor_data_alpaca_inst_{limit}_samples.json"
    out_file_path = os.path.join(out_dataset_dir, out_file_name)
    dataset_name = "GPUMODE/pytorch_scrape_inductor_data"


    # PyTorch Scrape
    # out_dataset_dir = "/matx/u/simonguo/triton_sft_data"
    # limit = 7500

    # out_file_name = f"pytorch_synthetic_data_alpaca_inst_{limit}_samples.json"
    # out_file_path = os.path.join(out_dataset_dir, out_file_name)
    # dataset_name = "simonguozirui/popcorn-synth-pytorch-triton"
    
    convert_dataset_from_huggingface(
        dataset_name=dataset_name,
        output_file=out_file_path,
        limit=limit
    )

    print(f"Converting dataset {dataset_name} to Alpaca format and saving to {out_file_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_name", default="GPUMODE/pytorch_scrape_inductor_data", 
#                         help="HuggingFace dataset name")
#     parser.add_argument("--output_file", default="pytorch_scrape_github_inductor_data_alpaca.json", 
#                         help="Output dataset file path")
#     parser.add_argument("--limit", type=int, help="Limit the number of examples (for debugging)", 
#                         default=None)
#     args = parser.parse_args()
    
#     # Convert the dataset to Alpaca format
#     convert_dataset_from_huggingface(
#         dataset_name=args.dataset_name,
#         output_file=args.output_file,
#         limit=args.limit
#     )