import json
import os
import random
from huggingface_hub import hf_hub_download

def download_and_merge_data():
    # Define dataset and file details
    dataset_name = "HANI-LAB/Med-REFL-DPO"
    files = ["Reasoning_Enhancement_Data.json", "Reflection_Enhancement_Data.json"]
    output_file = "Med-REFL_ALL.json"
    
    # Download files from Hugging Face
    data = []
    for file_name in files:
        file_path = hf_hub_download(
            repo_id=dataset_name,
            filename=file_name,
            repo_type="dataset"
        )
        # Read JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
            # Ensure file_data is a list
            if isinstance(file_data, list):
                data.extend(file_data)
            else:
                print(f"Warning: {file_name} is not a list, skipping.")
    
    # Shuffle the combined data
    random.shuffle(data)
    
    # Save merged data to output file
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Merged and shuffled data saved to {output_path}")

if __name__ == "__main__":
    download_and_merge_data()