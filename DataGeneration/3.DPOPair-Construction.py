"""
DPO Pair Text Generation Module

This script implements the final stage of a medical AI pipeline for generating
Direct Preference Optimization (DPO) training data. It converts structured DPO
reflection pairs into natural language training text with multiple output formats.

Key Features:
- Generates first-person conversational medical reasoning chains
- Creates educational reflections that identify and correct reasoning errors
- Supports both XML tags and Think-Conclusion output formats
- Applies text augmentation for diversity in training data
- Uses concurrent processing for efficient large-scale generation
- Integrates with local VLLM service for LLM inference
"""

import concurrent.futures
from tqdm import tqdm
import json
from openai import OpenAI
import time
import logging
import random
import os
from typing import List, Dict, Any

# Configure logging for the DPO text generation process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============================================================================
# 1. LLM INTERACTION MODULE
# =============================================================================

class VLLMChat:
    """
    Wrapper for interacting with local VLLM service for LLM inference.

    This class encapsulates the OpenAI-compatible API client for communicating
    with locally hosted VLLM models, providing a simple chat interface for
    generating medical reasoning text.
    """

    def __init__(
        self,
        # Use environment variables or default values for flexibility
        api_base: str = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1"),
        model_path: str = os.environ.get("VLLM_MODEL_PATH", "/path/to/your/model"),
        api_key: str = "EMPTY",
    ) -> None:
        """
        Initialize VLLM client with specified configuration.

        Args:
            api_base: Base URL for VLLM API endpoint
            model_path: Model identifier for VLLM service
            api_key: API key (typically "EMPTY" for local VLLM)
        """
        self.api_base = api_base
        self.model_path = model_path
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def chat(self, user_prompt: str, system_prompt: str, **kwargs) -> str:
        """
        Send chat completion request to VLLM model and return text response.

        Args:
            user_prompt: The user's input prompt
            system_prompt: System-level instructions for the model
            **kwargs: Additional parameters for the API call

        Returns:
            str: Model's text response or error information if failed
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,  # Moderate temperature for creative but consistent output
                max_tokens=4096,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"VLLM chat error: {e}")
            return f"Error: {e}"  # Return error info for debugging

# =============================================================================
# 2. PROMPT DEFINITION MODULE
# =============================================================================

# Prompt for converting structured reasoning steps into first-person conversational chain of thought
PROMPT_FIRST_THINK_COT = '''Create a conversational, first-person chain of thought that sounds like a biomedical student thinking aloud while solving a problem. Start with one of these natural conversation starters:
- "Okay, so I've got this question about..."
- "Well, let me see... this is asking about..."
- "Hmm, interesting... we're looking at a problem about..."

Then continue your thought process that:
1. Uses casual thinking-out-loud language throughout ("let me think...", "wait a minute...", "oh right...")
2. Maintains a biomedical student's perspective with appropriate terminology
3. Shows natural pauses and realizations ("Ah, now I see...")
4. Follows the logical steps from the original reasoning
5. Concludes with a casual but confident summary ("So yeah, putting it all together...")

Write as if you're recording your actual thought process.
'''

# Prompt for generating educational reflections that analyze reasoning errors
PROMPT_GENERATE_REFLECTION = """
You are a medical education assistant. Your task is to analyze clinical reasoning and provide an educational reflection.
You will receive:
[Question]: The clinical scenario.
[Public_Steps]: The common initial reasoning steps.
[Wrong_Steps]: The erroneous step or path taken.
[Right_Steps]: The correct approach that should have been taken.

Generate a concise [Reflection] that includes:
1. A clear identification of the flaws in the initial thinking.
2. Analysis of what key information was overlooked or misinterpreted.
3. Identification of any missing domain knowledge that led to the error.
4. A logical bridge explaining how recognizing these issues leads to the correct approach.

Keep your reflection focused on the critical analytical insights.
"""

# Prompt for combining all components into final DPO text with XML tags
PROMPT_COMBINE_WITH_REFLECTION = """
You will create a first-person chain of thought that demonstrates a natural problem-solving process, following a specific format.

Based on the provided:
[Question]
[First Think Solution] - The initial (flawed) thought process.
[Reflection] - An analysis of the errors.
[Correction] - The correct reasoning path.

Create a response with these exact sections:

<thinking>
{first_think_cot}
</thinking>

<reflection>
In this section, incorporate the content from [Reflection] naturally into a first-person voice. Explain where you went wrong, what you overlooked, and what new insight leads to a better solution.
</reflection>

<conclusion>
Synthesize your reflection and initial thinking to arrive at the correct solution based on [Correction]. Summarize your complete understanding and confidently present the final answer.
</conclusion>

Maintain a conversational, first-person voice throughout.
"""

# =============================================================================
# 3. TEXT PROCESSING AND AUGMENTATION MODULE
# =============================================================================

# Mapping of common phrases to alternatives for text augmentation
REPLACEMENT_MAP = {
    "Okay, so I've got this question about": ["Let's discuss a clinical scenario about", "Here's an interesting medical case about", "Now, I'm working through a clinical case where", "Okay, so I've got this question about"],
    "Wait a minute": ["Before we continue", "Wait a minute", "Hold on a second", "Let me see", "Hang on a moment"],
    "Ah, now I see": ["Oh, I get it now", "Aha, that makes sense", "OK, now it's clear to me", "Ah, that explains it"],
    "So yeah, putting it all together": ["So yeah, putting it all together", "Alright, so putting it all together", "Okay, to sum it up", "Well, wrapping it all up"],
    "let me think": ["let me process this", "let's see", "I'm considering", "let me figure this out", "let me think"]
}

# Reflection indicator phrases used in Think-Conclusion format
REFLECTION_FLAGS = [
    "Hold on, something feels off. Let me double-check.",
    "Wait, I think I missed something. Let me look again.",
    "Hmm, that doesn't seem right. I'll check my work.",
    "Hang on, let me make sure I got this right.",
    "Oh, wait, I need to review that again.",
    "Something's not adding up. Let me go over it.",
    "Hold up, I might've messed up. Let me check.",
    "Uh-oh, I feel like I got that wrong. I'll look it over.",
    "Just a sec, I need to revisit my answer.",
    "Wait a minute, I'm not sure about that. Let me confirm.",
    "Hmm, I'd better check that again, just in case.",
    "Oh, I think I need to go back and look at that.",
    "Hang on, let me see if I made a mistake.",
    "Wait, that feels wrong. I'll take another look.",
    "Hold on, I want to make sure I didn't screw up.",
    "Something's off. Let me review my thoughts.",
    "Just a moment, I need to double-check my answer.",
    "Hmm, I'm second-guessing myself. Let me check.",
    "Oh, I'd better go over that one more time.",
    "Wait, I'm not confident about that. Let me verify."
]

def apply_replacements(text: str) -> str:
    """
    Apply random replacements to specific phrases in text to increase diversity.

    Args:
        text: Input text to augment

    Returns:
        str: Text with random phrase replacements applied
    """
    for original, replacements in REPLACEMENT_MAP.items():
        if original in text:
            text = text.replace(original, random.choice(replacements), 1)  # Replace only once per phrase
    return text

def check_and_format_tags(text: str) -> str | None:
    """
    Check and format XML tags (thinking, reflection, conclusion) in generated text.

    Ensures all required tags are present and properly formatted with line breaks.

    Args:
        text: Generated text containing XML tags

    Returns:
        str | None: Formatted text if valid, None if missing required tags
    """
    required_tags = ["<thinking>", "</thinking>", "<reflection>", "</reflection>", "<conclusion>", "</conclusion>"]

    if not all(tag in text for tag in required_tags):
        logging.warning("Generated text is missing required tags.")
        return None  # Return None to indicate invalid format

    # Add proper line breaks around tags for better formatting
    text = text.replace("<thinking>", "\n<thinking>\n").replace("</thinking>", "\n</thinking>\n")
    text = text.replace("<reflection>", "\n<reflection>\n").replace("</reflection>", "\n</reflection>\n")
    text = text.replace("<conclusion>", "\n<conclusion>\n").replace("</conclusion>", "\n</conclusion>\n")
    return text.strip()

def transform_to_think_conclusion_format(content: str) -> tuple[bool, str]:
    """
    Transform XML tag format to Think-Conclusion format.

    Converts the structured XML format (thinking/reflection/conclusion tags)
    to a simpler format with ## Thinking and ## Conclusion headers,
    inserting random reflection flags between sections.

    Args:
        content: Text content with XML tags

    Returns:
        tuple[bool, str]: (success_status, transformed_content_or_error_message)
    """
    # Check for required tags and proper tag pairing
    required_pairs = [
        ('<thinking>', '</thinking>'),
        ('<conclusion>', '</conclusion>')
    ]

    # Check reflection tags for proper pairing
    if '<reflection>' in content and '</reflection>' not in content:
        return (False, "Missing closing </reflection> tag")
    if '</reflection>' in content and '<reflection>' not in content:
        return (False, "Missing opening <reflection> tag")

    # Check thinking and conclusion tags for proper pairing
    for open_tag, close_tag in required_pairs:
        if open_tag not in content or close_tag not in content:
            return (False, f"Missing {open_tag} or {close_tag} tag")
        if content.count(open_tag) != content.count(close_tag):
            return (False, f"Mismatched number of {open_tag} and {close_tag} tags")

    # Extract content from each section
    try:
        thinking_start = content.index('<thinking>') + len('<thinking>')
        thinking_end = content.index('</thinking>')
        thinking_content = content[thinking_start:thinking_end].strip()

        # Check if reflection section exists
        has_reflection = '<reflection>' in content and '</reflection>' in content
        reflection_content = ""
        if has_reflection:
            reflection_start = content.index('<reflection>') + len('<reflection>')
            reflection_end = content.index('</reflection>')
            reflection_content = content[reflection_start:reflection_end].strip()

        conclusion_start = content.index('<conclusion>') + len('<conclusion>')
        conclusion_end = content.index('</conclusion>')
        conclusion_content = content[conclusion_start:conclusion_end].strip()

        # Build new format content
        new_content = f"""## Thinking

{thinking_content}"""

        if has_reflection:
            random_flag = random.choice(REFLECTION_FLAGS)
            new_content += f"\n{random_flag}\n{reflection_content}"

        new_content += f"\n\n## Conclusion\n\n{conclusion_content}"

        return (True, new_content)
    except Exception as e:
        return (False, f"Error processing content: {str(e)}")

# =============================================================================
# 4. CORE PROCESSING LOGIC
# =============================================================================

def generate_dpo_pair_text(item: dict, qwen_instance: VLLMChat, output_format: str = "tags") -> dict | None:
    """
    Generate complete DPO text pairs for a single record from the pipeline.

    This function processes a single DPO reflection pair through multiple LLM
    calls to generate natural language training data with reasoning chains,
    educational reflections, and final conclusions.

    Args:
        item: Input data item containing question and reasoning steps
        qwen_instance: VLLM client instance for LLM inference
        output_format: Output format - "tags" for XML format, "think_conclusion" for simplified format

    Returns:
        dict | None: Processed item with r_chosen and r_rejected text fields, or None if failed
    """
    try:
        # --- Parse input data ---
        question = item["question"]
        spub = item["Spub"]  # Public reasoning steps
        pair_type = item["type"]

        # Determine error source based on pair type
        if pair_type == "mid-reasoning":
            serr_or_sorig = item["serr"]  # Error from reasoning process
        else:  # post-reasoning
            serr_or_sorig = item["Sorig"]  # Error from original answer

        snew_chosen = item["Snew_chosen"]  # Correct reasoning path (chosen)
        snew_rejected = item["Snew_rejected"]  # Alternative reasoning path (rejected)

        # --- Generate core components (3 parallel LLM calls for efficiency) ---

        # a) Generate initial erroneous thought process (first_think_cot)
        original_erroneous_path = f"{spub}\n{serr_or_sorig}".strip()
        user_prompt_cot = f"[Question]\n{question}\n[Chain of Thought]\n{original_erroneous_path}\n\nCreate your natural, spoken chain of thought:"
        first_think_cot = qwen_instance.chat(user_prompt_cot, PROMPT_FIRST_THINK_COT)
        first_think_cot = apply_replacements(first_think_cot)

        # b) Generate reflection text for "chosen" path
        user_prompt_ref_chosen = f"[Question]\n{question}\n[Public_Steps]\n{spub}\n[Wrong_Steps]\n{serr_or_sorig}\n[Right_Steps]\n{snew_chosen}\n\n[Reflection]\n"
        reflection_chosen = qwen_instance.chat(user_prompt_ref_chosen, PROMPT_GENERATE_REFLECTION)

        # c) Generate reflection text for "rejected" path
        reflection_rejected = None
        if snew_rejected:
            user_prompt_ref_rejected = f"[Question]\n{question}\n[Public_Steps]\n{spub}\n[Wrong_Steps]\n{serr_or_sorig}\n[Right_Steps]\n{snew_rejected}\n\n[Reflection]\n"
            reflection_rejected = qwen_instance.chat(user_prompt_ref_rejected, PROMPT_GENERATE_REFLECTION)

        # --- Combine components into final text ---

        # a) Combine to generate r_chosen
        correction_chosen = f"{spub}\n{snew_chosen}".strip()
        user_prompt_combine_chosen = f"[Question]\n{question}\n[First Think Solution]\n{original_erroneous_path}\n[Reflection]\n{reflection_chosen}\n[Correction]\n{correction_chosen}"
        r_chosen = qwen_instance.chat(
            user_prompt_combine_chosen,
            PROMPT_COMBINE_WITH_REFLECTION.format(first_think_cot=first_think_cot)
        )
        r_chosen = check_and_format_tags(r_chosen)

        # b) Combine to generate r_rejected
        r_rejected = None
        if reflection_rejected:
            correction_rejected = f"{spub}\n{snew_rejected}".strip()
            user_prompt_combine_rejected = f"[Question]\n{question}\n[First Think Solution]\n{original_erroneous_path}\n[Reflection]\n{reflection_rejected}\n[Correction]\n{correction_rejected}"
            r_rejected = qwen_instance.chat(
                user_prompt_combine_rejected,
                PROMPT_COMBINE_WITH_REFLECTION.format(first_think_cot=first_think_cot)
            )
            r_rejected = check_and_format_tags(r_rejected)

        # If either result is invalid, skip this data point
        if not r_chosen or not r_rejected:
            logging.warning(f"Failed to generate valid r_chosen or r_rejected for q_id {item.get('q_id')}. Skipping.")
            return None

        # --- Convert text based on output format ---
        if output_format == "think_conclusion":
            # Convert to Think-Conclusion format
            chosen_valid, r_chosen_transformed = transform_to_think_conclusion_format(r_chosen)
            rejected_valid, r_rejected_transformed = transform_to_think_conclusion_format(r_rejected)

            if not chosen_valid or not rejected_valid:
                logging.warning(f"Failed to transform content to think_conclusion format for q_id {item.get('q_id')}. Skipping.")
                return None

            r_chosen = r_chosen_transformed
            r_rejected = r_rejected_transformed
        # If "tags" format, keep original format unchanged

        # --- Return final result ---
        item['r_chosen'] = r_chosen
        item['r_rejected'] = r_rejected
        return item

    except Exception as e:
        logging.error(f"Error processing item for q_id {item.get('q_id')}: {e}", exc_info=True)
        return None

# =============================================================================
# 5. MAIN EXECUTION MODULE
# =============================================================================

def main_with_config(config):
    """
    Execute Step 3 of the pipeline using configuration object.

    Processes DPO reflection pairs from Step 2 output and generates final
    training text with specified output format using concurrent processing.

    Args:
        config: Configuration object containing all necessary parameters

    Returns:
        dict: Processing statistics including total tasks, processed count, success rate
    """
    # Extract configuration parameters
    input_path = config.step2_output_path
    output_path = config.step3_output_path
    output_format = config.dpo_format
    max_workers = config.dpo_max_workers

    logging.info(f"Starting DPO pair generation with format: {output_format}")
    logging.info(f"Input: {input_path}")
    logging.info(f"Output: {output_path}")

    # --- Initialize processing environment ---
    # Ensure output file is empty/doesn't exist
    if os.path.exists(output_path):
        os.remove(output_path)

    # Load tasks from input file
    with open(input_path, 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]

    # Create independent VLLM client instance for each worker thread
    qwen_instances = [VLLMChat() for _ in range(max_workers)]

    logging.info(f"Loaded {len(tasks)} tasks from {input_path}.")

    # --- Concurrent processing ---
    processed_count = 0
    skipped_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {}
        for i, item in enumerate(tasks):
            # Round-robin assignment of VLLM instances
            instance = qwen_instances[i % max_workers]
            future = executor.submit(generate_dpo_pair_text, item, instance, output_format)
            future_to_task[future] = item.get('q_id', 'Unknown')

        # Use tqdm to show progress
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Generating DPO Texts"):
            q_id = future_to_task[future]
            try:
                result = future.result()
                if result:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    processed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                logging.error(f"Task for q_id {q_id} generated an exception: {e}")
                skipped_count += 1

    # Log completion statistics
    logging.info(f"🎉 Step3 complete statistics:")
    logging.info(f"Processing complete. {processed_count} items processed, {skipped_count} items skipped.")
    logging.info(f"Final DPO pairs saved to {output_path}")

    return {
        'total_tasks': len(tasks),
        'processed_count': processed_count,
        'skipped_count': skipped_count,
        'success_rate': processed_count / len(tasks) if len(tasks) > 0 else 0.0
    }

def main():
    """
    Main entry point with command-line argument parsing.

    Supports both configuration file and command-line parameter modes
    for flexible execution of the DPO text generation pipeline.
    """
    import argparse

    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description='Generate DPO pairs with different output formats')
    parser.add_argument('--input_path', type=str,
                       default="path/to/your/dpo_reflection_pairs.jsonl",
                       help='Input file path from step 2 output')
    parser.add_argument('--output_path', type=str,
                       default="path/to/your/final_dpo_pairs_with_text.jsonl",
                       help='Output file path for final DPO pairs')
    parser.add_argument('--format', type=str, choices=['tags', 'think_conclusion'],
                       default='tags',
                       help='Output format: "tags" for <thinking><reflection><conclusion> format, "think_conclusion" for ## Thinking ## Conclusion format')
    parser.add_argument('--max_workers', type=int, default=10,
                       help='Number of concurrent workers')

    args = parser.parse_args()

    # Check for configuration file, use it if available, otherwise use command-line parameters
    try:
        from config import config
        print("✅ Running using configuration file for Step3")
        results = main_with_config(config)
    except ImportError:
        print("⚠️  Configuration file not found, using command-line parameters for Step3")

        # Create temporary configuration from command-line parameters
        class TempConfig:
            def __init__(self, args):
                self.step2_output_path = args.input_path
                self.step3_output_path = args.output_path
                self.dpo_format = args.format
                self.dpo_max_workers = args.max_workers

        config = TempConfig(args)
        results = main_with_config(config)

if __name__ == "__main__":
    main()