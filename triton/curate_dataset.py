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

from tqdm import tqdm
from datasets import load_dataset

TRITON_INSTRUCTION = "Convert this PyTorch neural network code to its optimized Triton implementation."

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
        data = data.select(range(limit))

    # Convert to Alpaca format
    # Apply limit if specified (for debugging)
    print(f"Dataset {dataset_name} contains {data.num_rows} rows, processing with limit {limit}")

    # final converted data
    converted_data = []

    for item in tqdm(data):
        import pdb; pdb.set_trace()
        entry_point = item["entry_point"]
        uuid = item["uuid"]
        converted_item = {
            "instruction": TRITON_INSTRUCTION,
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
    convert_dataset_from_huggingface(
        dataset_name="GPUMODE/pytorch_scrape_inductor_data",
        output_file="pytorch_scrape_github_inductor_data_alpaca.json",
        limit=5
    )

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