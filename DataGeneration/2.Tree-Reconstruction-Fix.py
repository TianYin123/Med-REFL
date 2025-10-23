# -*- coding: utf-8 -*-
"""
Tree-Reconstruction-Fix.py

This script implements solution tree reconstruction from medical question reasoning paths,
LLM-based error location in reasoning chains using medical expertise, and DPO (Direct
Preference Optimization) reflection pair generation. It processes the output from the
Tree-of-Thought reasoning pipeline to generate structured training pairs for medical AI.

Main Components:
1. LLM-based Error Locator: Identifies the first critical error in reasoning paths
2. Solution Tree Reconstruction: Builds enhanced solution trees with metadata
3. Value Calculation: Computes node values for tree analysis
4. DPO Pair Generation: Creates training pairs for preference optimization
5. Integration with OpenRouter API for structured LLM output

Author: Medical AI Research Pipeline
"""

import json
from collections import deque
import math
import os
import random
import sys
import itertools
import re
from openai import OpenAI


# ==============================================================================
# 1. LLM-BASED ERROR LOCATOR FOR REASONING PATHS
# ==============================================================================

def create_error_locator_schema():
    """
    Creates the JSON schema for structured output from the Error Locator LLM.

    This schema defines the expected format for LLM responses when identifying
    error steps in medical reasoning paths. It ensures consistent parsing of
    error identification results.

    Returns:
        dict: JSON schema with 'reasoning' and 'error_step' fields
    """
    return {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "A brief and precise explanation of why the selected step is the first critical error."
            },
            "error_step": {
                "type": "string",
                "description": "The number of the single incorrect step identified from the CANDIDATE list, e.g., '3' or 'Step3'."
            }
        },
        "required": ["reasoning", "error_step"]
    }

def normalize_step_identifier(error_step_str):
    """
    Normalizes step identifiers from LLM responses to consistent "StepX" format.

    Handles various formats that the LLM might return, such as "3", "Step3",
    "step 3", "#3", etc., and converts them all to the standardized "StepX" format.

    Args:
        error_step_str (str): Raw step identifier from LLM response

    Returns:
        str: Normalized step identifier in "StepX" format, or None if parsing fails
    """
    if not error_step_str:
        return None

    error_step_str = str(error_step_str).strip()

    # Try multiple pattern matches for different formats
    patterns = [
        r'^step\s*(\d+)$',        # "step 3", "step 12"
        r'^(\d+)$',               # "3", "12"
        r'ste?p?\s*(\d+)$',      # "step3", "step12" (allow lowercase)
        r'^#?(\d+)$',             # "#3", "#12"
    ]

    for pattern in patterns:
        match = re.match(pattern, error_step_str.lower())
        if match:
            step_num = match.group(1)
            return f"Step{step_num}"

    # If already in "Step" format, validate and return
    if re.match(r'^Step\d+$', error_step_str):
        return error_step_str

    # Handle "Step 3" format with space
    if re.match(r'^Step\s+(\d+)$', error_step_str):
        step_num = re.match(r'^Step\s+(\d+)$', error_step_str).group(1)
        return f"Step{step_num}"

    return None

def call_error_locator_llm(question, ground_truth, erroneous_path, candidate_steps):
    """
    Calls LLM to identify the first critical error in medical reasoning paths.

    Uses OpenRouter API with GPT-4 to analyze erroneous reasoning paths and
    identify the specific step that caused the final incorrect answer. The LLM
    receives the question, ground truth answer, full erroneous path, and a
    pre-selected list of candidate error steps.

    Args:
        question (str): The medical question text
        ground_truth (str): The correct answer for the question
        erroneous_path (str): The full text of the incorrect reasoning path
        candidate_steps (dict): Dictionary of candidate error steps with step keys as keys

    Returns:
        str: The identified error step key (e.g., "Step3") or None if identification fails
    """
    system_prompt = """You are an expert medical educator and a master of logical reasoning. Your task is to act as an "Error Locator" for a student's reasoning process on a USMLE-style question.

You will be given a medical question, the full reasoning path taken by a student which led to a WRONG answer, the actual correct answer, and a pre-selected list of CANDIDATE ERROR STEPS.

Your objective is to analyze the full path and identify the **single, first step** from the CANDIDATE list that is the primary reason for the final error. This could be a step that is medically inaccurate, contains a logical fallacy, or misinterprets the provided clinical data, causing the subsequent reasoning to be incorrect.

IMPORTANT: Return your answer as a valid step identifier from the CANDIDATE list exactly as shown."""

    # Format candidate steps for clearer display in prompt
    candidate_list_str = "\n".join([
        f"- **{key}**: {value[:150]}..." if len(value) > 150 else f"- **{key}**: {value}"
        for key, value in candidate_steps.items()
    ])

    user_prompt = f"""**[CONTEXT & DATA]**

**1. QUESTION:**
{question}

**2. GROUND TRUTH ANSWER:**
{ground_truth}

**3. FULL ERRONEOUS PATH:**
{erroneous_path}

**4. CANDIDATE ERROR STEPS:**
{candidate_list_str}

---
**INSTRUCTIONS:**
Please evaluate each step within the CANDIDATE list in the context of the FULL ERRONEOUS PATH and the GROUND TRUTH ANSWER.

Pinpoint the one step that represents the first critical deviation from a correct line of reasoning.

**RESPONSE FORMAT:**
- Return exactly the step identifier from the CANDIDATE list (e.g., "Step3", "Step1", etc.)
- DO NOT add any additional text or explanation in the final answer field
"""

    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", "your-openrouter-key-here"),
        base_url="https://openrouter.ai/api/v1"
    )

    try:
        response = client.chat.completions.create(
            model="qwen/qwen2.5-72b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_object",
            }
        )
        content = response.choices[0].message.content

        # Log raw response for debugging
        print(f"[DEBUG] LLM Raw Response: {content}")

        parsed_json = json.loads(content)

        # Normalize LLM returned step identifier
        raw_error_step = parsed_json.get("error_step", "")
        normalized_error_step = normalize_step_identifier(raw_error_step)

        print(f"[DEBUG] Raw error_step: '{raw_error_step}' -> Normalized: '{normalized_error_step}'")
        print(f"[DEBUG] Available candidate steps: {list(candidate_steps.keys())}")

        # Validate response format
        if normalized_error_step and normalized_error_step in candidate_steps:
            print(f"[SUCCESS] LLM Error Locator identified '{normalized_error_step}'. Reason: {parsed_json.get('reasoning', 'No reasoning provided')}")
            return normalized_error_step
        else:
            print(f"[WARNING] LLM returned invalid error_step: '{raw_error_step}' (normalized: '{normalized_error_step}')")
            print(f"[WARNING] Expected one of: {list(candidate_steps.keys())}")

            # Improved fallback strategy: try intelligent matching
            for candidate_key in candidate_steps.keys():
                if raw_error_step and raw_error_step.lower() in candidate_key.lower():
                    print(f"[FALLBACK] Matched '{raw_error_step}' to '{candidate_key}'")
                    return candidate_key

            # Final fallback: return first candidate step (return key, not value)
            first_candidate = next(iter(candidate_steps))
            print(f"[FALLBACK] Using first candidate: '{first_candidate}'")
            return first_candidate

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        print(f"[ERROR] Raw content: {content}")
        return next(iter(candidate_steps)) if candidate_steps else None
    except Exception as e:
        print(f"[ERROR] Error calling Error Locator LLM: {e}")
        return next(iter(candidate_steps)) if candidate_steps else None


# ==============================================================================
# 2. DATA READING AND PROCESSING FUNCTIONS
# ==============================================================================

def read_jsonl_to_list(file_path):
    """
    Reads JSONL file and converts to list of dictionaries with validation.

    Compatible with QM-ToT-VLLM-ToTRollout-Merged.py output format. Automatically
    maps 'ideal_answer' to 'ground_truth_answer' for consistency. Validates required
    fields and filters out invalid records.

    Args:
        file_path (str): Path to the JSONL file to read

    Returns:
        list: List of validated record dictionaries
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if line.strip():  # Skip empty lines
                record = json.loads(line)

                # Compatible with QM-ToT-VLLM-ToTRollout-Merged.py output format
                # Maps ideal_answer to ground_truth_answer
                if 'ideal_answer' in record and 'ground_truth_answer' not in record:
                    record['ground_truth_answer'] = record['ideal_answer']

                # Check required fields
                if 'solution' not in record or 'flag' not in record:
                    print(f"Warning: Record {i} missing 'solution' or 'flag'")
                    continue

                # Ensure record has q_id field
                if 'q_id' not in record:
                    print(f"Warning: Record {i} missing 'q_id'")
                    continue

                data.append(record)
    return data

def getjsonl(path):
    """
    Reads JSONL file and returns list of dictionaries.

    Simple utility function for reading JSONL files with error handling.

    Args:
        path (str): Path to the JSONL file

    Returns:
        list: List of parsed JSON objects
    """
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return data

def get_dev_data_for_options():
    """
    Loads development data to extract options information for questions.

    Retrieves additional data needed to reconstruct complete questions with
    their answer options. Uses environment variable for path configuration.

    Returns:
        list: List of development data records, or empty list if loading fails
    """
    dev_path = os.environ.get("DEV_DATA_PATH", "/path/to/dev/data.jsonl")
    try:
        dev_data = getjsonl(dev_path)
        return dev_data
    except Exception as e:
        print(f"Warning: Could not load dev data for options: {e}")
        return []

def reconstruct_full_question(record, dev_data):
    """
    Reconstructs complete question with options from record and development data.

    Combines the question text from the main record with answer options from
    development data to create a complete question including choices.

    Args:
        record (dict): Main record containing question text
        dev_data (list): Development data containing options information

    Returns:
        str: Complete question text with options, or original question if no match found
    """
    question = record.get("question", "")

    # If question already includes options, return as is
    if "the options are:" in question:
        return question

    # Otherwise, find corresponding options from dev_data and combine
    for dev in dev_data:
        dev_q = dev.get("question", "")
        dev_options = dev.get("options", "")
        # Check if questions match (compare without options portion)
        if question == dev_q:
            full_question = str(dev_q) + "\n" "the options are:" + str(dev_options)
            return full_question

    # If no match found, return original question
    return question

def extract_step_content(text, step_number=None):
    """
    Extracts step content or reasoning from solution text.

    Parses solution text to extract specific step content or the reasoning
    section. Handles both numbered steps (Step1, Step2, etc.) and the
    final Reason section.

    Args:
        text (str): Text to search for step content
        step_number (int, optional): Specific step number to extract. If None, extracts Reason section.

    Returns:
        str: Extracted content, or None if not found
    """
    # Split text into lines
    lines = text.splitlines()

    if step_number is not None:
        # Find specific step
        target = f"Step{step_number} is: "
        for line in lines:
            if target in line:
                # Extract content after the colon and trim whitespace
                content = line.split(target, 1)[1].strip()
                return content
    else:
        # Find Reason section - extract from "Reason:" to end of string
        reason_index = text.find("Reason:")
        if reason_index != -1:
            # Return from "Reason:" start to end of string content
            return text[reason_index:].strip()

    # If no matching content found, return None
    return None

def extract_actual_step_numbers(solution_text):
    """
    Extracts actual step numbers and content from solution text.

    Parses solution text to create a mapping of actual step numbers to their
    content. This helps maintain the original step numbering when reconstructing
    solution trees.

    Args:
        solution_text (str): Complete solution text with step markers

    Returns:
        dict: Mapping of step numbers to step content
    """
    step_mapping = {}
    lines = solution_text.splitlines()

    for line in lines:
        # Match various step formats
        step_match = re.match(r'^(Step\s*(\d+)\s*is:\s*)(.+)$', line.strip())
        if step_match:
            step_num = int(step_match.group(2))  # Extract numeric part
            step_content = step_match.group(3).strip()
            step_mapping[step_num] = step_content

    return step_mapping


# ==============================================================================
# 3. TREE CONSTRUCTION AND VALUE CALCULATION
# ==============================================================================

def get_backtrack_steps(leaf_node):
    """
    Calculates the number of steps to backtrack from a leaf node.

    Dynamically calculates backtrack steps based on the node's wrong_step_key
    position in the reasoning path. Used for determining how far back to go
    when finding alternative solution paths.

    Args:
        leaf_node (dict): Leaf node from solution tree

    Returns:
        int: Number of steps to backtrack (0 for correct paths)
    """
    if leaf_node['flag'] != -1:
        return 0  # Correct paths don't need backtracking

    path = get_path_to_root(leaf_node)
    total_steps = len(path) - 2  # Exclude root node and leaf node

    # Calculate actual backtrack steps = total steps - error step position + 1
    wrong_step_pos = leaf_node['wrong_step_key']
    backtrack_steps = total_steps - wrong_step_pos

    # Ensure backtrack steps are reasonable
    return max(0, min(backtrack_steps, total_steps))

def count_steps(text):
    """
    Counts the number of steps in solution text.

    Parses text to find the highest step number, which represents the total
    number of reasoning steps in the solution.

    Args:
        text (str): Solution text containing step markers

    Returns:
        int: Total number of steps found
    """
    lines = text.splitlines()
    step_count = 0
    for line in lines:
        if "Step" in line and " is: " in line:
            current_step = line.split("Step", 1)[1].split(" is:", 1)[0].strip()
            try:
                step_num = int(current_step)
                step_count = max(step_count, step_num)
            except ValueError:
                pass
    return step_count

def build_enhanced_solution_trees(branches_list, dev_data):
    """
    Builds enhanced solution trees with node IDs, parent references, and metadata.

    Creates hierarchical tree structures from reasoning branches, adding enhanced
    features like unique node IDs, parent references, depth tracking, and metadata
    storage for each node. Each tree represents all possible reasoning paths for
    a single medical question.

    Args:
        branches_list (list): List of reasoning branches for all questions
        dev_data (list): Development data for options information

    Returns:
        dict: Dictionary with question IDs as keys and enhanced trees as values
    """
    # Group branches by question ID
    questions_by_id = {}
    for branch in branches_list:
        q_id = branch['q_id']
        if q_id not in questions_by_id:
            questions_by_id[q_id] = []
        questions_by_id[q_id].append(branch)

    # Build solution tree for each question
    solution_trees = {}
    for q_id, branches in questions_by_id.items():
        # Reconstruct complete question (including options)
        question_text = reconstruct_full_question(branches[0], dev_data)

        # Create root node
        root_node = {
            'id': f"{q_id}_root",
            'type': 'question',
            'content': question_text,
            'q_id': q_id,
            'depth': 0,
            'children': {},     # Child node collection
            'parent': None,     # Root node has no parent
            'path': [],         # Path from root to this node
            'metadata': {}      # Expandable metadata storage
        }

        node_counter = 0  # Generate unique node IDs

        # Process each branch
        for branch in branches:
            current_node = root_node

            # Extract actual step mapping from solution text
            actual_step_mapping = extract_actual_step_numbers(branch.get('solution', ''))

            # Determine how many steps this branch has
            steps = []
            step_index = 1
            while f'step{step_index}' in branch:
                steps.append((f'step{step_index}', branch[f'step{step_index}']))
                step_index += 1

            # Extract reasoning (now processed as part of steps)
            reason = extract_step_content(branch['solution'])

            # Build branch path - fix: use actual step mapping
            for i, (step_key, step_value) in enumerate(steps):
                # Check if this content value is already a child of current node
                if step_value not in current_node['children']:
                    node_counter += 1
                    node_id = f"{q_id}_node{node_counter}"

                    # Fix: determine actual step number
                    actual_step_num = i + 1
                    # Try to find corresponding actual step number from solution text
                    for actual_num, actual_content in actual_step_mapping.items():
                        if actual_content == step_value:
                            actual_step_num = actual_num
                            break

                    # If last step, create leaf node
                    if i == len(steps) - 1:
                        new_node = {
                            'id': node_id,
                            'type': 'leaf',
                            'content': step_value,
                            'step_key': step_key,
                            'step_index': actual_step_num,  # Use actual step number
                            'depth': current_node['depth'] + 1,
                            'children': {},
                            'parent': current_node,
                            'path': current_node['path'] + [step_value],
                            'flag': branch['flag'],
                            'reason': reason,  # Store reason as leaf node attribute
                            'metadata': {'original_branch': branch},
                            'wrong_step_key': branch.get('wrong_step_key'),
                            'reason_stepkey': branch.get('reason_stepkey'),
                            'reason_reflection': branch.get('reason_reflection')
                        }
                    else:
                        # Create intermediate step node
                        new_node = {
                            'id': node_id,
                            'type': 'step',
                            'content': step_value,
                            'step_key': step_key,
                            'step_index': actual_step_num,  # Use actual step number
                            'depth': current_node['depth'] + 1,
                            'children': {},
                            'parent': current_node,
                            'path': current_node['path'] + [step_value],
                            'metadata': {}
                        }

                    # Add new node to current node's children
                    current_node['children'][step_value] = new_node

                # Move to next node to continue building
                current_node = current_node['children'][step_value]

        solution_trees[q_id] = root_node

    return solution_trees

def add_value_to_tree_nodes(enhanced_trees):
    """
    Adds value attributes to all nodes in enhanced solution trees and solution_value to leaf nodes.

    Calculates value scores for nodes based on the success/failure rates of their
    descendant leaf nodes. For internal nodes, value represents the normalized
    difference between correct and incorrect solutions. For leaf nodes, solution_value
    represents the average value of the path leading to that leaf.

    Args:
        enhanced_trees (dict): Enhanced solution trees dictionary

    Returns:
        dict: Enhanced trees with added value and solution_value attributes
    """
    def calculate_node_value(node):
        """Recursively calculates node value based on descendant leaves."""

        # If leaf node, use flag value directly
        if node['type'] == 'leaf':
            # Leaf node value is the flag value (1 or -1)
            node['value'] = float(node['flag'])
            return 1, node['flag'] == 1, node['flag'] == -1

        # Internal nodes need recursive calculation of all children
        total_leaves = 0
        positive_leaves = 0
        negative_leaves = 0

        # Process all child nodes
        for child in node['children'].values():
            leaves, positive, negative = calculate_node_value(child)
            total_leaves += leaves
            positive_leaves += positive
            negative_leaves += negative

        # Calculate current node value
        if total_leaves > 0:
            node['value'] = (positive_leaves - negative_leaves) / total_leaves
        else:
            # If no leaf nodes, set default value
            node['value'] = 0.0

        return total_leaves, positive_leaves, negative_leaves

    def add_solution_value_to_leafs(node, path_values=None, path_count=0):
        """Recursively adds solution_value attribute to leaf nodes."""
        if path_values is None:
            # Initialize empty path (excluding root node value)
            path_values = []
            path_count = 0

        # If current node is leaf node, calculate solution_value
        if node['type'] == 'leaf':
            if path_count > 0:  # Ensure path has nodes
                node['solution_value'] = sum(path_values) / path_count
            else:
                # Directly connected to root node leaf node, no intermediate nodes
                node['solution_value'] = 0.0
            return

        # Add current node value to path (root node and children below)
        if 'parent' in node and node['parent'] is not None:  # Determine non-root node
            current_path_values = path_values + [node['value']]
            current_path_count = path_count + 1
        else:
            # Root node not included in path
            current_path_values = []
            current_path_count = 0

        # Recursively process all child nodes
        for child in node['children'].values():
            add_solution_value_to_leafs(child, current_path_values, current_path_count)

    # Process each tree
    for q_id, tree in enhanced_trees.items():
        # First calculate all node values
        calculate_node_value(tree)

        # Then add solution_value to each leaf node
        add_solution_value_to_leafs(tree)

    return enhanced_trees


# ==============================================================================
# 4. TREE PRINTING AND UTILITY FUNCTIONS
# ==============================================================================

def print_solution_tree_with_values(enhanced_trees, q_id, include_stats=True, max_content_length=100):
    """
    Prints solution tree structure with values and node information.

    Displays the complete solution tree for a given question ID, including
    node values, solution values for leaf nodes, and optional statistics.
    Includes safe access to prevent crashes and content length limits.

    Args:
        enhanced_trees (dict): Enhanced solution trees dictionary
        q_id (str): Question ID to print
        include_stats (bool): Whether to include statistics (default: True)
        max_content_length (int): Maximum content display length (default: 100)
    """

    def print_tree_node(node, indent=0, max_content_length=100):
        """
        Recursively prints tree nodes with values and leaf node solution values.
        Internal helper function with safe access to prevent crashes.
        """
        # Safely get content, handle potential None or non-string types
        content = node.get('content', 'N/A')  # Use .get to provide default value
        if content is None:
            content = "[no content]"
        elif not isinstance(content, str):
             content = str(content)  # Ensure content is string

        # Truncate overly long content
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        # Safely get value, use NaN if missing
        value = node.get('value', float('nan'))
        # Check if NaN to decide how to format
        value_str = f"{value:.2f}" if not math.isnan(value) else "nan"

        # Print different formats based on node type
        node_type = node.get('type', 'unknown')  # Safely get type

        if node_type == 'question':
            print(' ' * indent + f"Question: {content} [value: {value_str}]")
        elif node_type == 'leaf':
            # Safely get flag, solution_value, reason, step_index
            flag = node.get('flag', 0)
            flag_text = "✓ correct" if flag == 1 else ("✗ error" if flag == -1 else "? unknown status")

            solution_value = node.get('solution_value', float('nan'))
            solution_value_str = f"{solution_value:.2f}" if not math.isnan(solution_value) else "nan"

            reason = node.get('reason', 'no reason provided')
            if not isinstance(reason, str):
                reason = str(reason)
            if len(reason) > max_content_length:
                reason = reason[:max_content_length] + "..."

            step_index = node.get('step_index', '?')  # Show '?' if missing

            print(' ' * indent + f"step{step_index} (leaf): {content} [{flag_text}, value: {value_str}, solution_value: {solution_value_str}]")
            print(' ' * (indent+4) + f"Reason: {reason}")
        elif node_type == 'step':  # Intermediate step node
            step_index = node.get('step_index', '?')
            print(' ' * indent + f"step{step_index}: {content} [value: {value_str}]")
        else:  # Unknown type node
             print(' ' * indent + f"unknown type node: {content} [value: {value_str}]")

        # Print child nodes, sort by content (key) to maintain consistent output order
        children_dict = node.get('children', {})
        if not isinstance(children_dict, dict):
             print(' ' * (indent + 4) + f"error: children should be dictionary, but got {type(children_dict)}")
             return  # Cannot process non-dictionary type children

        # Try to sort by key (content string)
        try:
            # Ensure key is string for sorting
            children_items = sorted(children_dict.items(), key=lambda item: str(item[0]))
        except TypeError:
            # If keys cannot be sorted (theoretically shouldn't happen as keys are step content strings), don't sort
            children_items = children_dict.items()

        for _, child in children_items:
             # Ensure child itself is dictionary before performing recursion
            if isinstance(child, dict):
                print_tree_node(child, indent + 4, max_content_length)  # Recursive call, pass parameter
            else:
                print(' ' * (indent + 4) + f"error: found non-dictionary type child {type(child)}")

    # --- Main function print_solution_tree_with_values logic start ---
    if q_id not in enhanced_trees:
        print(f"Error: Question ID '{q_id}' does not exist in enhanced tree data.")
        return

    # Get corresponding tree root node
    tree = enhanced_trees[q_id]
    if not isinstance(tree, dict):
        print(f"Error: Question ID '{q_id}' corresponding data is not a valid tree (dictionary type).")
        return

    # Safely get root node (question) content and score
    question_content = tree.get('content', 'N/A')
    if not isinstance(question_content, str):
        question_content = str(question_content)

    if len(question_content) > max_content_length:
        question_content = question_content[:max_content_length] + "..."

    root_value = tree.get('value', float('nan'))  # Safely get root node score
    root_value_str = f"{root_value:.2f}" if not math.isnan(root_value) else "nan"

    print("\n" + "="*80)
    print(f"Question ID: {q_id}")
    print(f"Question content: {question_content}")
    print(f"Question difficulty score (value): {root_value_str}")

    # --- Statistics info section ---
    if include_stats:
        total_solutions = 0
        correct_solutions = 0
        max_depth = 0

        # Use queue for breadth-first search (BFS) to traverse nodes for statistics, safer to prevent recursion depth issues
        nodes_to_process = deque()
        if isinstance(tree, dict):  # Ensure root node is dictionary
             nodes_to_process.append(tree)

        visited_ids = set()  # Used to prevent infinite loops in tree structure exceptions (like loops)

        processed_nodes_for_stats = 0  # Debug counter

        while nodes_to_process:
            current_node = nodes_to_process.popleft()

            # Basic check to ensure is dictionary
            if not isinstance(current_node, dict):
                continue

            # Check if already visited (if node has ID)
            node_id = current_node.get('id')
            if node_id:  # Only check when ID exists
                if node_id in visited_ids:
                    continue
                visited_ids.add(node_id)

            processed_nodes_for_stats += 1

            # Safely check node type and statistics
            node_type = current_node.get('type')
            if node_type == 'leaf':
                total_solutions += 1
                if current_node.get('flag') == 1:  # Safely access flag
                    correct_solutions += 1
                max_depth = max(max_depth, current_node.get('depth', 0))  # Safely access depth

            # Safely add child nodes to queue
            children = current_node.get('children', {})
            if isinstance(children, dict):  # Ensure children is dictionary
                for child in children.values():
                    if isinstance(child, dict):  # Ensure child is dictionary
                        nodes_to_process.append(child)

        # Print statistics results
        print("-"*80)
        print(f"Statistics:")
        # print(f"(processed {processed_nodes_for_stats} nodes for statistics)")  # Optional debug info
        print(f"Total solutions (leaf nodes): {total_solutions}")
        if total_solutions > 0:
            correct_pct = (correct_solutions / total_solutions * 100)
            incorrect_pct = ((total_solutions - correct_solutions) / total_solutions * 100)
            print(f"Correct solutions: {correct_solutions} ({correct_pct:.1f}%)")
            print(f"Error solutions: {total_solutions - correct_solutions} ({incorrect_pct:.1f}%)")
        else:
             print("No valid leaf nodes found.")
        print(f"Maximum solution steps (maximum leaf depth): {max_depth}")
        print("-"*80)
    # --- Statistics info section end ---

    print("\nSolution tree structure:")
    # Start recursive printing from root node
    if isinstance(tree, dict):
        print_tree_node(tree, max_content_length=max_content_length)  # Pass parameter
    else:
        print("Error: Tree root node is not a valid dictionary.")

    print("\n" + "="*80)

def get_path_to_root(leaf_node):
    """
    Gets complete path from leaf node to root node (including both start and end points).

    Traverses up the tree from a leaf node to the root, building the complete
    reasoning path. Useful for analysis and path reconstruction.

    Args:
        leaf_node (dict): Leaf node from solution tree

    Returns:
        list: Path from leaf to root (reversed order), or empty list if node is None
    """
    path = []
    current = leaf_node

    while current is not None:
        path.append(current)
        current = current.get('parent')

    # Path is from leaf to root, needs to be reversed
    return list(reversed(path))

def format_path_to_solution_text(path):
    """
    Converts path to text format solution: "Step1: xxx\n Step2: xxx\n ... \n Reason: xxx".

    Formats the reasoning path as readable text with step numbers and reasoning.
    Excludes the root/question node and includes the final reasoning if available.

    Args:
        path (list): List of nodes representing the reasoning path

    Returns:
        str: Formatted solution text
    """
    steps_text = []

    # Skip first node in path (question node/root node)
    for i, node in enumerate(path[1:], 1):
        # Only process non-question nodes
        if node['type'] != 'question':
            step_index = node.get('step_index', i)  # Use node's original step index (if exists)
            steps_text.append(f"Step{step_index} is: {node['content']}")

    # Add reasoning, only if leaf node has reasoning
    reason = path[-1].get('reason')
    if reason:
        steps_text.append(f"{reason}")

    return "\n".join(steps_text)

def format_partial_path(path, start_index=1, end_index=None):
    """
    Formats a portion of the path as text.

    Extracts and formats a sub-segment of the reasoning path for analysis
    or display purposes.

    Args:
        path (list): Complete path list
        start_index (int): Starting index (default: 1)
        end_index (int, optional): Ending index (default: None for end of path)

    Returns:
        str: Formatted partial path text
    """
    if end_index is None:
        end_index = len(path)

    steps_text = []
    # Iterate over specified range of nodes in path
    for i, node in enumerate(path[start_index:end_index], 1):
        if node['type'] != 'question':  # Skip question node
            step_index = node.get('step_index', i)  # Use node's original step index (if exists)
            steps_text.append(f"Step{step_index} is: {node['content']}")

    return "\n".join(steps_text)

def format_partial_path_2(path, start_index=1, end_index=None):
    """
    Formats a portion of the path as text with reasoning included.

    Similar to format_partial_path but includes reasoning for leaf nodes.
    Used for generating complete solution text segments.

    Args:
        path (list): Complete path list
        start_index (int): Starting index (default: 1)
        end_index (int, optional): Ending index (default: None for end of path)

    Returns:
        str: Formatted partial path text with reasoning
    """
    if end_index is None:
        end_index = len(path)

    steps_text = []
    # Iterate over specified range of nodes in path
    for i, node in enumerate(path[start_index:end_index], 1):
        if node['type'] != 'question':  # Skip question node
            step_index = node.get('step_index', i)  # Use node's original step index (if exists)
            steps_text.append(f"Step{step_index} is: {node['content']}")
            if node['type'] == 'leaf':
                steps_text.append(node.get("reason"))
    return "\n".join(steps_text)

def format_remaining_path(path, start_node_index):
    """
    Formats the portion of the path after a specified starting node.

    Used to generate the remaining reasoning path after an error point,
    typically for alternative solution generation.

    Args:
        path (list): Complete path
        start_node_index (int): Starting node index (format from node after this)

    Returns:
        str: Formatted text for remaining path
    """
    steps_text = []

    # Start from node after start_node_index
    for node in path[start_node_index+1:]:
        if node['type'] != 'question':  # Skip question node
            step_index = node.get('step_index')  # Use node's original step index
            steps_text.append(f"Step{step_index} is: {node['content']}")

    # Add reasoning, only if leaf node has reasoning
    reason = path[-1].get('reason')
    if reason:
        steps_text.append(f"{reason}")

    return "\n".join(steps_text)

def get_backtracked_node(leaf_node, n):
    """
    Gets the parent node n steps back from a leaf node.

    Used to find the backtrack point for generating alternative solutions.
    Returns both the backtracked node and the error step node.

    Args:
        leaf_node (dict): Leaf node from solution tree
        n (int): Number of steps to backtrack

    Returns:
        tuple: (backtracked_node, wrong_step_node, full_path) or (None, None, path) if backtrack exceeds range
    """
    path = get_path_to_root(leaf_node)

    # Calculate actual backtrackable length
    # Backtrackable length = path length - 2 (excluding root node and leaf node itself)
    backtrackable_length = len(path) - 2

    # If backtrack steps greater than or equal to backtrackable length, return None
    if n >= backtrackable_length:
        return None, None, path

    # Calculate backtracked node index
    # Leaf node index is path.length-1, after backtracking n steps index is path.length-1-n
    backtracked_index = len(path) - 1 - n

    # Get backtracked node
    backtracked_node = path[backtracked_index]

    # Get error step node (node after backtracked node)
    wrong_step_node = path[backtracked_index + 1] if backtracked_index + 1 < len(path) else None

    return backtracked_node, wrong_step_node, path

def find_all_correct_solutions(node):
    """
    Finds all paths from given node that reach flag=1 leaf nodes.

    Performs breadth-first search to find all correct solution paths starting
    from a given node. Used for identifying successful reasoning trajectories.

    Args:
        node (dict): Starting node for search

    Returns:
        list: List of paths, each path is a list of nodes ending in a correct leaf
    """
    if node is None:
        return []

    correct_solutions = []
    queue = deque([(node, [node])])  # Each item includes (current node, path to reach this node)

    while queue:
        current_node, current_path = queue.popleft()

        # If leaf node and flag=1, add to correct solutions
        if current_node.get('type') == 'leaf' and current_node.get('flag') == 1:
            correct_solutions.append(current_path)
            continue

        # Expand to child nodes
        for child in current_node.get('children', {}).values():
            new_path = current_path + [child]
            queue.append((child, new_path))

    return correct_solutions

def find_all_wrong_solutions(node):
    """
    Finds all paths from given node that reach flag=-1 leaf nodes.

    Performs breadth-first search to find all incorrect solution paths starting
    from a given node. Used for identifying failed reasoning trajectories.

    Args:
        node (dict): Starting node for search

    Returns:
        list: List of paths, each path is a list of nodes ending in an incorrect leaf
    """
    if node is None:
        return []

    wrong_solutions = []
    queue = deque([(node, [node])])  # Each item includes (current node, path to reach this node)

    while queue:
        current_node, current_path = queue.popleft()

        # If leaf node and flag=-1, add to wrong solutions
        if current_node.get('type') == 'leaf' and current_node.get('flag') == -1:
            wrong_solutions.append(current_path)
            continue

        # Expand to child nodes
        for child in current_node.get('children', {}).values():
            new_path = current_path + [child]
            queue.append((child, new_path))

    return wrong_solutions

def find_all_leaf_nodes(node, flag_value):
    """
    Finds all leaf nodes with a specific flag value from a given start node.

    Generalized function to find leaf nodes based on their flag value.
    Used for both correct (flag=1) and incorrect (flag=-1) solution searches.

    Args:
        node (dict): Starting node for search
        flag_value (int): Flag value to search for (1 for correct, -1 for incorrect)

    Returns:
        list: List of paths ending in leaf nodes with specified flag value
    """
    leaves = []
    queue = deque([(node, [node])])
    while queue:
        current_node, current_path = queue.popleft()
        if current_node.get('type') == 'leaf' and current_node.get('flag') == flag_value:
            leaves.append(current_path)
        for child in current_node.get('children', {}).values():
            new_path = current_path + [child]
            queue.append((child, new_path))
    return leaves


# ==============================================================================
# 5. DPO PAIR GENERATION AND ANALYSIS
# ==============================================================================

def calculate_action_value(right_step_value, wrong_step_value, right_solution_value, wrong_solution_value, remaining_value):
    """
    Calculates action value using weights from the research paper (0.4, 0.2, 0.4).

    Implements the action value calculation formula from Appendix E.3 of the paper.
    This metric evaluates the quality improvement when replacing an incorrect step
    with a correct one in the reasoning path.

    Args:
        right_step_value (float): Value of the correct replacement step
        wrong_step_value (float): Value of the original incorrect step
        right_solution_value (float): Solution value of the correct path
        wrong_solution_value (float): Solution value of the incorrect path
        remaining_value (float): Average value of the remaining path

    Returns:
        float: Calculated action value score
    """
    # Weights from Appendix E.3
    lambda1, lambda2, lambda3 = 0.4, 0.2, 0.4

    # Calculate deltas for improvement
    step_delta = right_step_value - wrong_step_value
    solution_delta = right_solution_value - wrong_solution_value

    action_value = lambda1 * step_delta + lambda2 * solution_delta + lambda3 * remaining_value
    return action_value

def analyze_and_sort_alternatives(backtracked_node, original_wrong_step_node, original_wrong_leaf):
    """
    Finds and ranks all correct and incorrect alternative paths from a backtrack point.

    Analyzes all possible solution paths from the backtrack node, calculates action
    values for each, and sorts them by quality. Returns separate lists for correct
    (chosen) and incorrect (rejected) alternatives.

    Args:
        backtracked_node (dict): Node to backtrack to for alternative generation
        original_wrong_step_node (dict): The original incorrect step node
        original_wrong_leaf (dict): The original incorrect leaf node

    Returns:
        tuple: (right_solutions, wrong_solutions) - sorted lists of alternative paths
    """
    correct_paths = find_all_leaf_nodes(backtracked_node, 1)
    wrong_paths = find_all_leaf_nodes(backtracked_node, -1)

    wrong_step_value = original_wrong_step_node.get('value', 0.0)
    wrong_solution_value = original_wrong_leaf.get('solution_value', 0.0)

    # Process and rank right solutions
    right_solutions = []
    for path in correct_paths:
        if len(path) > 1:
            right_first_node = path[1]  # First step after backtrack point
            right_leaf = path[-1]
            path_values = [n.get('value', 0.0) for n in path[1:]]  # Values of remaining path
            remaining_value = sum(path_values) / len(path_values) if path_values else 0.0

            action_value = calculate_action_value(
                right_step_value=right_first_node.get('value', 0.0),
                wrong_step_value=wrong_step_value,
                right_solution_value=right_leaf.get('solution_value', 0.0),
                wrong_solution_value=wrong_solution_value,
                remaining_value=remaining_value
            )
            right_solutions.append({
                "action_value": action_value,
                "solution_value": right_leaf.get('solution_value', 0.0),
                "path": path,
                "content": format_partial_path_2(path, 1)  # Content from child of backtrack node onwards
            })
    right_solutions.sort(key=lambda x: x['action_value'], reverse=True)

    # Process and rank wrong solutions (hard negatives)
    wrong_solutions = []
    for path in wrong_paths:
        # Exclude the original path itself
        if path[-1]['id'] == original_wrong_leaf['id']:
            continue
        if len(path) > 1:
            wrong_first_node = path[1]
            wrong_leaf = path[-1]
            path_values = [n.get('value', 0.0) for n in path[1:]]
            remaining_value = sum(path_values) / len(path_values) if path_values else 0.0

            action_value = calculate_action_value(
                right_step_value=wrong_first_node.get('value', 0.0),  # 'right' here means the new step
                wrong_step_value=wrong_step_value,
                right_solution_value=wrong_leaf.get('solution_value', 0.0),
                wrong_solution_value=wrong_solution_value,
                remaining_value=remaining_value
            )
            wrong_solutions.append({
                "action_value": action_value,
                "path": path,
                "content": format_partial_path_2(path, 1)
            })
    # Sort by action_value descending to get the most plausible ("best") wrong alternatives
    wrong_solutions.sort(key=lambda x: x['action_value'], reverse=True)

    return right_solutions, wrong_solutions

def create_dpo_pairs(q_id, question_text, Spub, serr_or_sorig_content, right_solutions, wrong_solutions, n, m, pair_type):
    """
    Creates n x m DPO pairs from the top chosen and rejected solutions.

    Generates training pairs for Direct Preference Optimization by combining
    top-n correct solutions with top-m incorrect solutions. Creates Cartesian
    product for comprehensive coverage.

    Args:
        q_id (str): Question identifier
        question_text (str): Complete question text
        Spub (str): Public steps before the error point
        serr_or_sorig_content (str): Error step or original solution content
        right_solutions (list): List of correct alternative solutions
        wrong_solutions (list): List of incorrect alternative solutions
        n (int): Number of top correct solutions to use
        m (int): Number of top incorrect solutions to use
        pair_type (str): Type of DPO pair ("mid-reasoning" or "post-reasoning")

    Returns:
        list: List of DPO pair dictionaries
    """
    pairs = []
    top_chosen = right_solutions[:n]
    top_rejected = wrong_solutions[:m]

    if not top_chosen or not top_rejected:
        return []

    for chosen_path_data, rejected_path_data in itertools.product(top_chosen, top_rejected):
        pair = {
            "q_id": q_id,
            "question": question_text,
            "type": pair_type,
            "Spub": Spub,
            "Snew_chosen": chosen_path_data['content'],
            "Snew_rejected": rejected_path_data['content']
        }
        if pair_type == "mid-reasoning":
            pair["serr"] = serr_or_sorig_content
        else:  # post-reasoning
            pair["Sorig"] = serr_or_sorig_content
        pairs.append(pair)
    return pairs

def process_tree(q_id, tree, ground_truth_map, n, m, k_post, k_candidates):
    """
    Main processing function for a single question's tree.

    Orchestrates the complete analysis pipeline: finds errors, locates them with
    LLM, analyzes alternatives, and generates all DPO pairs for a single
    question's solution tree.

    Args:
        q_id (str): Question identifier
        tree (dict): Enhanced solution tree for the question
        ground_truth_map (dict): Mapping of question IDs to correct answers
        n (int): Number of top correct solutions for DPO pairs
        m (int): Number of top incorrect solutions for DPO pairs
        k_post (int): Number of top solutions for post-reasoning pairs
        k_candidates (int): Number of candidate steps for LLM error location

    Returns:
        list: All generated DPO pairs for this question
    """
    all_pairs = []
    # 1. Find all incorrect trajectories (leaves) in the tree
    incorrect_leaves_paths = find_all_leaf_nodes(tree, -1)

    if not incorrect_leaves_paths:
        return []

    print(f"\n--- Processing Q_ID: {q_id} ({len(incorrect_leaves_paths)} incorrect paths) ---")

    for path_idx, full_path in enumerate(incorrect_leaves_paths):
        wrong_leaf = full_path[-1]

        # 2. Dynamic Error Location
        intermediate_steps = [node for node in full_path[1:]]  # Exclude root
        if not intermediate_steps:
            print(f"[WARNING] No intermediate steps found for path {path_idx}, skipping")
            continue

        # Select candidates for the LLM based on lowest v_step
        intermediate_steps.sort(key=lambda node: node.get('value', 0.0))
        candidate_nodes = intermediate_steps[:k_candidates]

        # Use actual step numbers to build candidate dictionary
        candidate_steps_dict = {}
        for node in candidate_nodes:
            step_key = f"Step{node.get('step_index', 'unknown')}"
            candidate_steps_dict[step_key] = node['content']

        if not candidate_steps_dict:
            print(f"[WARNING] No candidate steps found for path {path_idx}, skipping")
            continue

        print(f"[DEBUG] Path {path_idx}: Selected {len(candidate_steps_dict)} candidates")
        print(f"[DEBUG] Candidates: {list(candidate_steps_dict.keys())}")

        erroneous_path_text = format_path_to_solution_text(full_path)
        ground_truth_answer = ground_truth_map.get(q_id, "Unknown")

        # Call LLM to find the error step's key, e.g., "Step3"
        error_step_key_str = call_error_locator_llm(
            tree['content'], ground_truth_answer, erroneous_path_text, candidate_steps_dict
        )

        if not error_step_key_str:
            print(f"[ERROR] LLM returned None for path {path_idx}, skipping")
            continue

        # Improve error node finding logic
        error_node = None
        for node in full_path:
            node_step_key = f"Step{node.get('step_index', 'unknown')}"
            if node_step_key == error_step_key_str:
                error_node = node
                break

        if not error_node:
            print(f"[ERROR] Could not find error step '{error_step_key_str}' in path for q_id {q_id}")
            print(f"[ERROR] Available steps in path: {[f' Step{node.get('step_index')}' for node in full_path if node.get('type') != 'question']}")
            continue

        if not error_node.get('parent'):
            print(f"[ERROR] Error node has no parent for q_id {q_id}, skipping")
            continue

        # 3. Get backtrack node and public steps
        backtracked_node = error_node.get('parent')
        error_node_index_in_path = full_path.index(error_node)
        Spub = format_partial_path(full_path, 1, error_node_index_in_path)

        print(f"[SUCCESS] Found error step '{error_step_key_str}' at depth {error_node.get('depth')}")
        print(f"[DEBUG] Backtrack node depth: {backtracked_node.get('depth')}")

        # 4. Find and rank all alternatives from the backtrack point
        right_solutions, wrong_solutions = analyze_and_sort_alternatives(backtracked_node, error_node, wrong_leaf)

        if not right_solutions or not wrong_solutions:
            print(f"[WARNING] Could not find sufficient correct/wrong alternatives for q_id {q_id}. Skipping.")
            continue

        print(f"[DEBUG] Found {len(right_solutions)} right and {len(wrong_solutions)} wrong alternatives")

        # 5. Generate pairs based on error type (leaf vs. intermediate)
        if error_node['type'] == 'leaf':
            # --- Handle Post-Reasoning as per new request ---
            right_solutions.sort(key=lambda x: x['solution_value'], reverse=True)
            top_k_for_post = right_solutions[:k_post]
            if top_k_for_post:
                chosen_for_post = random.choice(top_k_for_post)
                post_pair = {
                    "q_id": q_id,
                    "question": tree['content'],
                    "type": "post-reasoning",
                    "Spub": Spub,
                    "Sorig": error_node['content'] + "\n" + (error_node.get('reason') or ""),
                    "Snew_chosen": chosen_for_post['content'],
                    "Snew_rejected": None  # Post-reasoning can be a single chosen path vs the original error
                }
                all_pairs.append(post_pair)

            # --- Also generate Mid-Reasoning pairs from the leaf error ---
            # Re-sort by action_value for mid-reasoning logic
            right_solutions.sort(key=lambda x: x['action_value'], reverse=True)
            mid_pairs = create_dpo_pairs(
                q_id, tree['content'], Spub, error_node['content'],
                right_solutions, wrong_solutions, n, m, "mid-reasoning"
            )
            all_pairs.extend(mid_pairs)
        else:  # Error is an intermediate step
            mid_pairs = create_dpo_pairs(
                q_id, tree['content'], Spub, error_node['content'],
                right_solutions, wrong_solutions, n, m, "mid-reasoning"
            )
            all_pairs.extend(mid_pairs)

    return all_pairs


# ==============================================================================
# 6. MAIN EXECUTION AND CONFIGURATION
# ==============================================================================

def save_to_jsonl(data, output_path):
    """
    Saves a list of dictionaries to a JSONL file.

    Writes DPO pairs to output file in JSONL format with proper encoding
    for international characters.

    Args:
        data (list): List of dictionaries to save
        output_path (str): Path to output file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} DPO pairs to: {output_path}")

def main_with_config(config):
    """
    Main execution function using configuration object.

    Orchestrates the complete pipeline: data loading, tree construction,
    value calculation, and DPO pair generation. Uses configuration object
    for paths and parameters.

    Args:
        config: Configuration object with paths and parameters

    Returns:
        dict: Results summary with statistics
    """
    # Configuration paths
    input_file = config.step1_output_path
    output_dir = config.step2_output_dir
    output_file = config.step2_output_path

    # DPO parameters
    N_CHOSEN = config.N_CHOSEN
    M_REJECTED = config.M_REJECTED
    K_POST_REASONING = config.K_POST_REASONING
    K_CANDIDATES_FOR_LLM = config.K_CANDIDATES_FOR_LLM

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading and Preprocessing ---
    # Compatible with QM-ToT-VLLM-ToTRollout-Merged.py output format
    # Automatically maps ideal_answer to ground_truth_answer
    all_records = read_jsonl_to_list(input_file)
    ground_truth_map = {
        record['q_id']: record.get('ground_truth_answer', record.get('ideal_answer', 'Answer Not Provided'))
        for record in all_records
    }
    print(f"Read {len(all_records)} records from {input_file}")

    # Load development data for options information
    print("Loading dev data for options information...")
    dev_data = get_dev_data_for_options()
    print(f"Loaded {len(dev_data)} dev records")

    # Data validation and statistics
    valid_records = 0
    invalid_records = 0
    mixed_q_ids = set()

    for record in all_records:
        if record.get('flag') in [1, -1] and record.get('solution'):
            valid_records += 1
            mixed_q_ids.add(record.get('q_id'))
        else:
            invalid_records += 1

    print(f"Valid records: {valid_records}, Invalid records: {invalid_records}")
    print(f"Unique q_ids: {len(mixed_q_ids)}")

    # Extract steps from solution fields
    print("Extracting steps from solution fields...")
    extracted_count = 0
    for line in all_records:
        solution = line.get("solution", "")
        if solution:
            num_steps = count_steps(solution)
            for i in range(1, num_steps + 1):
                step = extract_step_content(solution, i)
                if step is not None:
                    line[f"step{i}"] = step
            extracted_count += 1

    print(f"Extracted steps from {extracted_count} records")

    # --- Tree Building and Value Calculation (Step 2) ---
    print("Building and analyzing solution trees...")

    # Debug: check if some example questions include options
    if all_records:
        sample_record = all_records[0]
        original_q = sample_record.get('question', '')[:100]
        reconstructed_q = reconstruct_full_question(sample_record, dev_data)[:100]
        print(f"Sample original question: {original_q}...")
        print(f"Sample reconstructed question: {reconstructed_q}...")
        print(f"Contains options: {'the options are:' in reconstructed_q}")

    enhanced_trees = build_enhanced_solution_trees(all_records, dev_data)
    enhanced_trees_with_values = add_value_to_tree_nodes(enhanced_trees)
    print(f"Built {len(enhanced_trees_with_values)} trees.")

    # --- DPO Pair Generation (Step 3) ---
    final_dpo_pairs = []
    for q_id, tree in enhanced_trees_with_values.items():
        pairs = process_tree(
            q_id, tree, ground_truth_map,
            n=N_CHOSEN, m=M_REJECTED, k_post=K_POST_REASONING, k_candidates=K_CANDIDATES_FOR_LLM
        )
        final_dpo_pairs.extend(pairs)

    # --- Save Final Results ---
    if final_dpo_pairs:
        save_to_jsonl(final_dpo_pairs, output_file)
        print(f"\n🎉 Step2 complete statistics:")
        print(f"Successfully generated {len(final_dpo_pairs)} DPO pairs!")

        # Statistics different type pairs
        mid_reasoning_pairs = [p for p in final_dpo_pairs if p.get('type') == 'mid-reasoning']
        post_reasoning_pairs = [p for p in final_dpo_pairs if p.get('type') == 'post-reasoning']
        print(f"Mid-reasoning pairs: {len(mid_reasoning_pairs)}")
        print(f"Post-reasoning pairs: {len(post_reasoning_pairs)}")

        return {
            'total_pairs': len(final_dpo_pairs),
            'mid_reasoning_pairs': len(mid_reasoning_pairs),
            'post_reasoning_pairs': len(post_reasoning_pairs),
            'trees_built': len(enhanced_trees_with_values),
            'valid_records': valid_records
        }
    else:
        print("❌ No DPO pairs were generated.")
        return {
            'total_pairs': 0,
            'mid_reasoning_pairs': 0,
            'post_reasoning_pairs': 0,
            'trees_built': len(enhanced_trees_with_values),
            'valid_records': valid_records
        }


# ==============================================================================
# 7. SCRIPT EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    # Check if configuration file exists, use it if available, otherwise use default configuration
    try:
        from config import config
        print("✅ Running using configuration file Step2")
        results = main_with_config(config)
    except ImportError:
        print("⚠️  Configuration file not found, using default configuration for Step2")

        # Default configuration
        class DefaultConfig:
            def __init__(self):
                self.step1_output_path = os.environ.get("STEP1_OUTPUT_PATH", "/path/to/step1/output.jsonl")
                self.step2_output_dir = os.environ.get("STEP2_OUTPUT_DIR", "/path/to/step2/output")
                self.step2_output_path = os.path.join(self.step2_output_dir, 'dpo_reflection_pairs.jsonl')
                self.N_CHOSEN = 2
                self.M_REJECTED = 2
                self.K_POST_REASONING = 6
                self.K_CANDIDATES_FOR_LLM = 4

        config = DefaultConfig()
        results = main_with_config(config)