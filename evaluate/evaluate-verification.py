import os
from tqdm import tqdm
import json
from openai import OpenAI
import concurrent.futures
from time import sleep
import glob
import logging

class HTTPRequestFilter(logging.Filter):
    """Custom log filter to filter out logs containing HTTP-related keywords."""
    def filter(self, record):
        msg = record.getMessage()
        return not any(keyword in msg for keyword in ["HTTP Request", "POST", "http://"])

def setup_logging(folder_path: str) -> None:
    """Configure logging to write to a log file in the specified folder and disable unnecessary HTTP request logs."""
    log_file = os.path.join(folder_path, "evaluation_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    # Disable debug logs for all modules that might generate HTTP logs
    for logger_name in [
        "openai", "openai._base_client", "httpx", "httpcore", 
        "urllib3", "http.client", "requests"
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False  # Prevent logs from propagating upwards
    # Add the custom filter to the root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(HTTPRequestFilter())

def getjsonl(path: str) -> list:
    """Read and parse a JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                line = line.strip()
                if line:
                    json_obj = json.loads(line)
                    data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue  # Skip the problematic line
    return data

def chat(
    user_prompt: str,
    system_prompt: str
):
    client = OpenAI(
    api_key="Your ChatGPT API", 
    base_url="Your ChatGPT URL"
    )
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    return response.choices[0].message.content

def extract_answer(text: str) -> str:
    """Extract the answer letter from the text."""
    VALID_ANSWERS = {'A', 'B', 'C', 'D', 'E', 'G', 'H', 'None'}
    for line in text.splitlines():
        if "Answer:" in line:
            answer = line.split("Answer:")[1].strip()
            if answer in VALID_ANSWERS:
                return answer
    return "fail"

def process_single_question(question_data, option_data, system_prompt, index, file_name):
    """Process a single question and return the result."""
    reason = question_data["reason"]
    ideal_answer = question_data["ideal_answer"]
    option = option_data["options"]
    reason = reason + "\nOptions:" + str(option)
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            resp = chat(user_prompt=reason, system_prompt=system_prompt)
            ans = extract_answer(resp)
            
            if ans == "fail":
                retry_count += 1
                reason = (reason + "<correct>\n" + 
                         "The right format is : Answer:A/B/C/D/E/None, remember just one of these \n" +
                         "But last time you output a wrong format:" + "{" + resp + "}" + 
                         "\nNow extract with the right format again.</correct>")
                continue
            
            return {
                "index": index,
                "file_name": file_name,
                "ideal_answer": ideal_answer,
                "model_answer": ans,
                "is_correct": ans == ideal_answer,
                "is_valid": ans != "None",
            }
            
        except Exception as e:
            retry_count += 1
            sleep(1)
    
    return {
        "index": index,
        "file_name": file_name,
        "ideal_answer": ideal_answer,
        "model_answer": "error",
        "is_correct": False,
        "is_valid": False,
        "error": "Exceeded maximum retries"
    }

def process_batch(solutions, options, system_prompt, start_idx, batch_size, file_name):
    """Process a batch of questions."""
    batch_results = []
    end_idx = min(start_idx + batch_size, len(solutions))
    
    process_params = []
    for i in range(start_idx, end_idx):
        if i < len(solutions) and i < len(options):
            process_params.append((solutions[i], options[i], system_prompt, i, file_name))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(process_single_question, *params) for params in process_params]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                batch_results.append(result)
            except Exception as e:
                print(f"Error processing question: {str(e)}")
                batch_results.append({
                    "index": -1,
                    "file_name": file_name,
                    "ideal_answer": "Unknown",
                    "model_answer": "error",
                    "is_correct": False,
                    "is_valid": False,
                    "error": str(e)
                })
    
    return batch_results

def main():
    SYSTEM_PROMPT = """You are tasked with extracting and categorizing the chosen option from a medical reasoning text. 

Given a text containing:
A detailed reasoning/explanation and the options to choose

Your task is to:
1. Analyze the reasoning text
2. Identify the explicitly stated chosen answer (usually indicated by phrases like "Therefore, the answer is" or "the correct answer is")
3. Extract only the letter corresponding to the chosen option
4. If the reasoning text didn't clearly indicate the answer it chooses, return with 'None' 

Rules:
- Only extract the single letter option (A, B, C, D, or E)
- Ignore any additional text or explanation
- Include only the chosen option or 'None' in the final output. 
- None means the reasoning text does not show a final choice. 
- The output format should be Answer:X where X is the letter of the chosen option or 'None'

Examples:
"...Therefore, the answer is D: 'Decrease production of gastrin'." → 
Answer:D
"...The correct answer is 3 ... Options:'A':5,'B':3 " → 
Answer:B
"\\boxed{C}" → 
Answer:C
"""
    
    folder_path = "results/example/huatuo-o1-Med-REFL"
    option_path = "data/MedQA_USLME_test.jsonl"
    # Configure logging
    setup_logging(folder_path)
    
    # Read options
    options = getjsonl(option_path)
    print(f"Data Path: {folder_path}\nOptions Path: {option_path}")
    logging.info(f"Loading options from {option_path}")
    
    
    # Get all JSONL files
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    if not jsonl_files:
        logging.error("No JSONL files found in the specified folder")
        print("Error: No JSONL files found in the specified folder")
        return
    
    batch_size = 40
    file_results = {}
    
    # Process each JSONL file
    for file_path in jsonl_files:
        file_name = os.path.basename(file_path)
        logging.info(f"Processing file: {file_name}")
        print(f"Processing file: {file_name}")
        
        solutions = getjsonl(file_path)
        all_results = []
        
        # Process questions in batches
        for start_idx in tqdm(range(0, len(solutions), batch_size), desc=f"Processing {file_name}"):
            batch_results = process_batch(solutions, options, SYSTEM_PROMPT, start_idx, batch_size, file_name)
            all_results.extend(batch_results)
        
            # Calculate current accuracy
            correct = sum(1 for r in all_results if r["is_correct"])
            incorrect = sum(1 for r in all_results if r["is_valid"] and not r["is_correct"])
            
            if (correct + incorrect) > 0:
                accuracy = correct / (correct + incorrect)
                print(f"Current Accuracy: {accuracy:.4f}")
        
        # Calculate final statistics
        correct = sum(1 for r in all_results if r["is_correct"])
        incorrect = sum(1 for r in all_results if r["is_valid"] and not r["is_correct"])
        invalid = sum(1 for r in all_results if not r["is_valid"])
        
        accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
        correct_indices = [r["index"] for r in all_results if r["is_correct"]]
        incorrect_indices = [r["index"] for r in all_results if r["is_valid"] and not r["is_correct"]]
        invalid_indices = [r["index"] for r in all_results if not r["is_valid"]]
        
        file_results[file_name] = {
            "Accuracy": accuracy,
            "Correct": correct,
            "Incorrect": incorrect,
            "Invalid": invalid,
            "Correct_Indices": correct_indices,
            "Incorrect_Indices": incorrect_indices,
            "Invalid_Indices": invalid_indices
        }
        
        # Log results for this file
        logging.info(f"Results for {file_name}:")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Correct: {correct}")
        logging.info(f"Incorrect: {incorrect}")
        logging.info(f"Invalid: {invalid}")
        logging.info(f"Correct Indices: {correct_indices}")
        logging.info(f"Incorrect Indices: {incorrect_indices}")
        logging.info(f"Invalid Indices: {invalid_indices}")
    
    # Calculate average, max, and min accuracy
    accuracies = [res["Accuracy"] for res in file_results.values()]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    max_accuracy_file = max(file_results.items(), key=lambda x: x[1]["Accuracy"])[0] if file_results else "N/A"
    min_accuracy_file = min(file_results.items(), key=lambda x: x[1]["Accuracy"])[0] if file_results else "N/A"
    
    # Print and log the summary
    print("\nFinal Statistics:")
    logging.info("\nSummary:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    logging.info(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"File with Highest Accuracy: {max_accuracy_file} (Accuracy: {file_results[max_accuracy_file]['Accuracy']:.4f})")
    logging.info(f"File with Highest Accuracy: {max_accuracy_file} (Accuracy: {file_results[max_accuracy_file]['Accuracy']:.4f})")
    print(f"File with Lowest Accuracy: {min_accuracy_file} (Accuracy: {file_results[min_accuracy_file]['Accuracy']:.4f})")
    logging.info(f"File with Lowest Accuracy: {min_accuracy_file} (Accuracy: {file_results[min_accuracy_file]['Accuracy']:.4f})")
    
    # Print and log the accuracy for all files
    print("\nAccuracy for all files:")
    logging.info("\nAccuracy for all files:")
    for file_name, results in file_results.items():
        print(f"{file_name}: {results['Accuracy']:.4f}")
        logging.info(f"{file_name}: {results['Accuracy']:.4f}")

if __name__ == "__main__":
    main()