import os
import json
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# --- Configuration Constants ---
# Model and Adapter Paths
BASE_MODEL_PATH = "FreedomIntelligence/HuatuoGPT-o1-8B"
ADAPTER_PATH = "HANI_LAB/Med-REFL-huatuo-o1"
ADAPTER_NAME = "med_refl_adapter"

# Data and Output Paths
DATA_FILE_PATH = "evaluate/data/MedQA_USLME_test.jsonl"
OUTPUT_FOLDER = "evaluate/results/example/huatuo-o1-Med-REFL"

# Inference Parameters
NUM_RUNS = 3
MAX_NEW_TOKENS = 8192
REPETITION_PENALTY = 1.1

# System prompt for the model
SYSTEM_PROMPT = """You are a helpful medical expert specializing in USMLE exam questions, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Your responses will be used for research purposes only, so please have a definite answer.
Provide your response in the following JSON format:
{"reason": "Step-by-step explanation of your thought process","answer": "Chosen answer from the given options"}
"""


# --- Data Handling and Formatting Functions ---

def load_jsonl(file_path: str) -> list:
    """Loads data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} contains invalid JSON.")
    return data

def write_to_jsonl(data: dict, file_path: str):
    """Appends a dictionary to a JSONL file."""
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")

def format_options(options_dict: dict) -> str:
    """Formats a dictionary of options into a multi-line string."""
    return '\n'.join([f'{key}: {value}' for key, value in options_dict.items()])

def create_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    """Creates a formatted prompt string for the model."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )


# --- Custom Sampling Distribution Functions for Temperature ---

def cdf(x: float) -> float:
    """
    Cumulative Distribution Function (CDF) for a custom piecewise probability distribution.
    This function defines the probability distribution for generating temperature values.
    """
    if x <= 0.2:
        return 0.0
    if 0.2 < x <= 0.6:
        # Integral of f(t) = 0.5 + 2.5*(t-0.2) from 0.2 to x
        return 0.5 * (x - 0.2) + 1.25 * (x - 0.2)**2
    if 0.6 < x <= 0.8:
        # Integral from 0.2 to 0.6, plus integral from 0.6 to x
        # Value at x=0.6 is 0.4
        return 0.4 + 1.5 * (x - 0.6)
    if 0.8 < x <= 1.0:
        # Integral from 0.2 to 0.8, plus integral from 0.8 to x
        # Value at x=0.8 is 0.7
        return 0.7 + 1.5 * (x - 0.8) - 1.25 * (x - 0.8)**2
    return 1.0 # for x > 1.0

def inverse_cdf(p: float) -> float:
    """
    Inverse CDF (Quantile Function) using binary search.
    Used for generating a random number that follows the custom distribution.
    """
    if p <= 0:
        return 0.2
    if p >= 1:
        return 1.0
    
    low, high = 0.2, 1.0
    while high - low > 1e-10:
        mid = (low + high) / 2
        if cdf(mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2

def generate_temperature_sample() -> float:
    """Generates a random temperature value based on the custom CDF."""
    random_probability = np.random.uniform(0, 1)
    return inverse_cdf(random_probability)


# --- Custom Sampling Function for Top-P ---

def generate_top_p_sample() -> float:
    """
    Generates a random top_p value from a weighted choice of two intervals.
    - 30% chance to be in [0.2, 0.6)
    - 70% chance to be in [0.6, 1.0)
    """
    intervals = [(0.2, 0.6), (0.6, 1.0)]
    weights = [0.3, 0.7]
    chosen_interval_index = np.random.choice([0, 1], p=weights)
    low, high = intervals[chosen_interval_index]
    return np.random.uniform(low, high)


# --- Main Execution ---

def main():
    """Main function to run the inference process."""
    # Ensure the output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Initialize the vLLM engine with LoRA enabled
    print("Initializing vLLM engine...")
    llm = LLM(model=BASE_MODEL_PATH, enable_lora=True, max_lora_rank=64)
    print("Engine initialized.")

    # Load and prepare prompts once
    print(f"Loading data from {DATA_FILE_PATH}...")
    qa_data = load_jsonl(DATA_FILE_PATH)
    if not qa_data:
        print("No data loaded. Exiting.")
        return

    print("Preparing prompts...")
    prompts = []
    for item in tqdm(qa_data, desc="Formatting Prompts"):
        question_text = item.get("question", "")
        options_dict = item.get("options", {})
        
        user_prompt = (
            f"Here is the USMLE question:\n{question_text}\n\n"
            f"The options are:\n{format_options(options_dict)}"
        )
        prompts.append(create_chat_prompt(SYSTEM_PROMPT, user_prompt))

    # --- Main inference loop ---
    for run_number in range(1, NUM_RUNS + 1):
        print(f"\n--- Starting Run {run_number}/{NUM_RUNS} ---")

        # Generate random sampling parameters for this run
        temp = round(generate_temperature_sample(), 2)
        top_p = round(generate_top_p_sample(), 2)
        
        print(f"Generated parameters: Temperature={temp}, Top-P={top_p}")

        # Set sampling parameters and output path for this run
        sampling_params = SamplingParams(
            temperature=temp,
            top_p=top_p,
            max_tokens=MAX_NEW_TOKENS,
            repetition_penalty=REPETITION_PENALTY
        )
        
        output_filename = f"huatuo_o1_Med-REFL_{run_number-1}.jsonl"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Clear the output file if it exists from a previous run
        if os.path.exists(output_path):
            os.remove(output_path)

        # Generate outputs using the vLLM engine
        lora_request = LoRARequest(ADAPTER_NAME, 1, ADAPTER_PATH)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        
        # Process and write results
        print(f"Writing results to {output_path}...")
        for original_item, model_output in zip(qa_data, outputs):
            result_text = model_output.outputs[0].text
            result_data = {
                "reason": result_text,
                "ideal_answer": original_item.get("answer_idx"),
                "temperature": temp,
                "top_p": top_p
            }
            write_to_jsonl(result_data, output_path)
            
        print(f"Completed Run {run_number}/{NUM_RUNS}.")

if __name__ == "__main__":
    main()
