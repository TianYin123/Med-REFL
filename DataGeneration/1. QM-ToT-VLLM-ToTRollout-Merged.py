"""
Medical Question Processing with Tree-of-Thought (ToT) Reasoning System

This module implements a comprehensive medical AI pipeline for processing clinical exam questions
using Tree-of-Thought reasoning methodology. It integrates multiple LLM APIs (Doubao, OpenRouter,
local VLLM) with concurrent processing capabilities to generate step-by-step analytical solutions
for medical questions.

Key Features:
- Multi-step medical question analysis with ToT reasoning
- Integration with multiple LLM services (Doubao, OpenRouter, local VLLM)
- JSON schema validation for structured outputs
- Concurrent processing for scalability
- Error handling with fallback mechanisms
- Step-by-step reasoning with evaluation and continuation logic
"""

import requests
import json
from tqdm import tqdm
from collections import Counter
import pickle
from time import sleep
from volcenginesdkarkruntime import Ark
from openai import OpenAI
import concurrent.futures
import threading
from typing import Dict, List, Any, Optional
import queue
import signal
import sys
import os

# ===================================================================
# LLM API Client Functions
# ===================================================================

def chat(
    user_prompt: str,
    system_prompt: str
) -> str:
    """
    Send a chat request to the Doubao model using volcenginesdkarkruntime.

    Args:
        user_prompt (str): The user input prompt
        system_prompt (str): The system prompt defining the model's behavior

    Returns:
        str: The model's response content
    """
    # Initialize the Doubao client with API key from environment
    client = Ark(api_key=os.environ.get("ARK_API_KEY", "your-api-key-here"))

    # Create the chat completion request
    response = client.chat.completions.create(
        model="ModelName",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )

    # Return the message content
    return response.choices[0].message.content

def chat_max(
    user_prompt: str,
    system_prompt: str,
    json_schema: dict = None
) -> dict:
    """
    Call OpenRouter API with structured JSON output support.

    This function provides structured output capabilities using JSON schema validation
    through the OpenRouter API service.

    Args:
        user_prompt (str): User input prompt
        system_prompt (str): System prompt defining model behavior
        json_schema (dict, optional): JSON schema for structured output validation

    Returns:
        dict: Parsed JSON response content

    Raises:
        ValueError: If model response cannot be parsed as JSON
    """
    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", "your-openrouter-key-here"),
        base_url="https://openrouter.ai/api/v1"
    )

    # Build request parameters
    request_params = {
        "model": "openai/gpt-5-nano",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    # Add structured output configuration if JSON schema is provided
    if json_schema:
        request_params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": json_schema
            }
        }

    # Send request and parse response
    response = client.chat.completions.create(**request_params)
    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError("Model response cannot be parsed as JSON, please check JSON Schema or model output.")

# ===================================================================
# JSON Schema Definitions for Structured LLM Outputs
# ===================================================================

# Schema for generating reasoning steps in Tree-of-Thought analysis
STEPS_SCHEMA = {
    "type": "object",
    "properties": {
        "step1": {
            "type": "string",
            "description": "First analysis step"
        },
        "step2": {
            "type": "string",
            "description": "Second analysis step"
        },
        "step3": {
            "type": "string",
            "description": "Third analysis step"
        }
    },
    "required": ["step1", "step2", "step3"],
    "additionalProperties": False
}

# Schema for determining if reasoning is sufficient to answer the question
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "Detailed explanation of the decision"
        },
        "decision": {
            "type": "string",
            "description": "yes or no, indicating if sufficient to answer the question"
        },
        "answer": {
            "type": "string",
            "description": "A/B/C/D/E or None, indicating the selected answer"
        }
    },
    "required": ["reason", "decision", "answer"],
    "additionalProperties": False
}

# Schema for evaluating reasoning quality and convincingness
CONVINCING_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "Explanation of the reasoning process evaluation"
        },
        "convincing?": {
            "type": "string",
            "description": "yes or no, indicating if reasoning is convincing"
        }
    },
    "required": ["reason", "convincing?"],
    "additionalProperties": False
}

# ===================================================================
# Local VLLM Client Integration
# ===================================================================

class VLLMChat:
    """
    Client for interacting with locally hosted VLLM models.

    This class provides a simplified interface for communicating with VLLM inference servers
    running locally, typically used for faster inference with local models.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8001/v1",
        model_path: str = os.environ.get("VLLM_MODEL_PATH", "/path/to/your/model"),
        api_key: str = "EMPTY",
    ) -> None:
        """
        Initialize VLLMChat client.

        Args:
            api_base (str): Base URL for VLLM API endpoint
            model_path (str): Path or identifier for the model
            api_key (str): API key (typically "EMPTY" for local VLLM)
        """
        self.api_base = api_base
        self.model_path = model_path
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )

    def chat(
        self,
        user_prompt: str,
        system_prompt: str,
        **kwargs
    ) -> str:
        """
        Interact with VLLM model and return response content.

        Args:
            user_prompt (str): User input prompt
            system_prompt (str): System prompt defining model behavior
            **kwargs: Additional parameters for the API call

        Returns:
            str: Model response content
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **kwargs
            )
            content = response.choices[0].message.content
            return content
        except json.JSONDecodeError:
            return {"error": "Cannot parse JSON response"}
        except Exception as e:
            return {"error": str(e)}

# ===================================================================
# Ollama Integration (Alternative Local Model Service)
# ===================================================================

# Default Ollama API endpoint
url_generate = "http://localhost:11434/api/generate"

def collectPrompt(system_prompt: str, user_prompt: str) -> str:
    """
    Combine system and user prompts into Ollama-compatible format.

    Args:
        system_prompt (str): System prompt content
        user_prompt (str): User prompt content

    Returns:
        str: Formatted prompt string for Ollama API
    """
    prompt=f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
    return prompt

class ollamaChat:
    """
    Client for interacting with Ollama local model service.

    Provides integration with Ollama for running local models with structured output support.
    """

    def __init__(self, url: str = url_generate, model: str = "Med-Refl:latest", format: str = "json") -> None:
        """
        Initialize Ollama chat client.

        Args:
            url (str): Ollama API endpoint URL
            model (str): Model name to use
            format (str): Output format (typically "json" for structured output)
        """
        self.url = url
        self.model = model
        self.format = format

    def getResp(self, data: dict) -> str:
        """
        Send request to Ollama API and get response.

        Args:
            data (dict): Request data payload

        Returns:
            str: Model response or error message
        """
        try:
            response = requests.post(self.url, json=data)
            # Check response status code
            response.raise_for_status()  # Raise HTTPError for request failures
            response_dict = response.json()  # Parse JSON response
            return response_dict.get("response", "Json format return error")  # Use .get() to avoid KeyError
        except requests.exceptions.RequestException as e:
            return f"An error occurred: {e}"

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send chat request to Ollama model.

        Args:
            system_prompt (str): System prompt content
            user_prompt (str): User prompt content

        Returns:
            str: Model response
        """
        prompt = collectPrompt(system_prompt, user_prompt)
        data = {
            "model": self.model,
            "prompt": prompt,
            "format": self.format,
            "stream": False
        }
        return self.getResp(data)

    def awakeollama(self, model: str, time: str = "12h") -> None:
        """
        Keep Ollama model awake for specified duration.

        Args:
            model (str): Model name to keep awake
            time (str): Duration to keep model loaded
        """
        data = {
            "model": model,
            "keep_alive": time
        }
        print(self.getResp(data))

# ===================================================================
# Prompt Templates for Medical Question Processing
# ===================================================================

def first_prompt_tot() -> str:
    """
    Generate prompt template for initial Tree-of-Thought step generation.

    Returns:
        str: Prompt template for generating three different initial analytical approaches
    """
    prompt ='''
You are a medical expert specializing in clinical medical exam questions. Your task is to suggest three different INITIAL analytical steps for approaching a given medical question.
IMPORTANT RULES:
1. This is an exam scenario - all patient information is ONLY in the question
2. Provide ONLY the FIRST step for each different approach
3. Each approach should start from a unique analytical perspective
4. Keep steps specific and actionable
Input:
Question: [clinical medical question]
# Your initial approaches could focus on:
# 1. Basic science perspective (anatomy/physiology)
# 2. Clinical presentation analysis
# 3. Pathological process understanding
# 4. Different diagnostic frameworks
Present your response in JSON format:
{
  "step1": "first analytical step based on the key information provided in the question",
  "step2": "Second analytical step based on the key information provided in the question",
  "step3": "Third analytical step based on the key information provided in the question"
}
Remember: Each step should be distinct, clear, and represent only the STARTING point of different analytical approaches.
'''
    return prompt

def ifproblemSolving() -> str:
    """
    Generate prompt template for evaluating if current reasoning is sufficient.

    Returns:
        str: Prompt template for determining if enough information exists to answer
    """
    prompt='''
You are a medical expert specializing in United States Medical License exam questions. You will be provided with a medical question, answer options, previous problem-solving steps, the current step, and relevant context from a medical textbook. Your task is to determine if the given information is sufficient to answer the question.
Remember : All relevant information about the patients and medical examine results has been attached to the questions.
Input:
1. Question: [A medical question from the US Medical License exam]
2. Options: [List of possible answer choices]
3. History: [Previous problem-solving steps conducted]
4. Step: [Current problem-solving step]
Analyze the provided information carefully. Then, respond with the following:
1. Reason: Explain your thought process in determining whether the given information is sufficient to answer the question. Consider how the previous steps (history), current step, relate to the question and options.
2. Decision: Based on your analysis, decide if the information is enough to choose an answer.
   - If the information is sufficient, write "yes".
   - If the information is insufficient, write "no".
3. Answer:
   - If your decision is "yes", provide the correct option from the given choices.
   - If your decision is "no", write "None".
Present your response in JSON format as follows:
{
  "reason": "Your detailed explanation",
  "decision": "yes" or "no",
  "answer": "Chosen option or None"
}
Ensure your reasoning is clear, concise, and directly related to the medical question at hand. Take into account the progression of steps and how the current information builds upon or relates to the previous steps in reaching a conclusion.
'''
    return prompt

def generateStepNoContent() -> str:
    """
    Generate prompt template for generating subsequent reasoning steps.

    Returns:
        str: Prompt template for creating next analytical steps without new patient data
    """
    prompt='''You are a medical expert specializing in clinical medical exam questions. Your task is to suggest the next three analytical steps based on a given medical question and previous problem-solving steps.
IMPORTANT RULES:
1. This is an exam scenario - all patient information is ONLY in the question
2. Do not suggest new tests, examinations, or patient interactions
3. Focus on analyzing and interpreting existing information only
Input:
Question: [clinical medical question]
History: [Previous steps]
LastStep: [Most recent step]
Your analysis should focus on:
1. Interpreting provided symptoms and test results
2. Developing differential diagnoses from given data
3. Evaluating treatment options based on known information
4. Applying medical knowledge to existing findings
Present your response in JSON format:
{
  "step1": "First analytical next step",
  "step2": "Second analytical next step",
  "step3": "Third analytical next step"
}
Ensure each step is specific, analytical, and based solely on information already provided in the question.
'''
    return prompt

def ifconvincingprompt() -> str:
    """
    Generate prompt template for evaluating reasoning quality.

    Returns:
        str: Prompt template for assessing if solution reasoning is convincing
    """
    prompt = '''
You are a medical education expert specializing in clinical medical exam preparation. Given a clinical medical question and its proposed solution steps, evaluate the quality and logic of the reasoning process.
First, carefully read the clinical medical question and the provided solution steps. Then, analyze the solution considering these key aspects:
1. Clarity: Are the steps clearly articulated and easy to follow?
2. Logical flow: Does each step naturally lead to the next?
3. Medical accuracy: Are the medical concepts and relationships correctly applied?
4. Completeness: Does the reasoning address all key elements of the question?
5. Evidence-based: Are the conclusions supported by the given information?
Provide your evaluation in JSON format with two fields:
- "reason": A explanation or comment of why the reasoning is effective or ineffective, highlighting specific strengths or weaknesses
- "convincing?": "yes" if the reasoning is clear, logical, and sufficient to answer the question; "no" if there are significant gaps or flaws in the reasoning
Your evaluation should focus on the quality of the reasoning process rather than just the correctness of the final answer.
'''
    return prompt

def complete_reasoning_prompt() -> str:
    """
    Generate prompt template for completing reasoning at maximum depth.

    Returns:
        str: Prompt template for final reasoning completion when max depth is reached
    """
    prompt = '''
You are a medical expert specializing in clinical medical exam questions. You have been provided with a medical question, answer options, and a series of analytical steps that have reached the maximum allowed reasoning depth.

Your task is to follow the previous steps in History, use one step(long or short is fine) to complete the reasoning and get the answer

Input:
1. Question: [A medical question from the clinical exam]
2. Options: [List of possible answer choices A/B/C/D/E]
3. History: [Previous analytical steps conducted, ending at maximum depth]

Instructions:
1. Review all the previous analytical steps carefully
2. Continue the logical reasoning process to reach a definitive conclusion
3. Clearly state your final answer choice from the given options

Output format:
Continue Reasoning: ...

Final Answer: [Choose from A/B/C/D/E]

Remember: Base your reasoning solely on the information provided in the question and the analytical work already completed. Do not introduce new external information or suggest additional tests/examinations.
'''
    return prompt

# ===================================================================
# Core Medical Question Processing Functions
# ===================================================================

def firststepGenToT(question: str) -> Dict[str, str]:
    """
    Generate initial Tree-of-Thought reasoning steps for a medical question.

    This function implements a multi-tier fallback strategy:
    1. Try local VLLM model first
    2. Use Doubao API for JSON format correction
    3. Fall back to OpenRouter with structured output
    4. Return default steps if all else fails

    Args:
        question (str): The medical question to analyze

    Returns:
        Dict[str, str]: Dictionary containing three initial reasoning steps
    """
    ques = f'''
    the question is :{question}
    '''
    chat_client = VLLMChat()

    # Primary VLLM call without retry mechanism
    try:
        resp = chat_client.chat(system_prompt=first_prompt_tot(), user_prompt=ques)
    except Exception as e:
        print(f"VLLM call failed: {e}")
        resp = None

    # Use Doubao API to rewrite/correct as proper JSON format
    if resp:
        doubao_prompt = f"""Please convert the following content to correct JSON format:
Original content: {resp}

Please return in the following JSON format:
{{
  "step1": "first analytical step",
  "step2": "second analytical step",
  "step3": "third analytical step"
}}

Note: Only return JSON format results, do not include any other text."""
    else:
        doubao_prompt = f"""Please generate three different initial analytical steps based on the following question:
Question: {question}

Please return in the following JSON format:
{{
  "step1": "first analytical step",
  "step2": "second analytical step",
  "step3": "third analytical step"
}}

Note: Only return JSON format results, do not include any other text."""

    # Doubao retry mechanism with 3 attempts, adding format guidance after each failure
    for retry in range(3):
        try:
            doubao_resp = chat(user_prompt=doubao_prompt, system_prompt="You are a JSON format conversion expert, only return valid JSON format content. Just convert the format, nothing more.")
            steps = json.loads(doubao_resp)
            return steps
        except json.JSONDecodeError as e:
            if retry < 2:
                # Add format guidance to original prompt
                doubao_prompt = (doubao_prompt + "<correct>\n" +
                               "The right format is : {\"step1\": \"step1\", \"step2\": \"step2\", \"step3\": \"step3\"}\n" +
                               "But last time you output a wrong format:" + "{" + doubao_resp + "}" +
                               "\nNow extract with the right format again.</correct>")
                continue
            else:
                # After 3 failures, use chat_max with structured output
                try:
                    max_resp = chat_max(
                        user_prompt=doubao_prompt,
                        system_prompt="You are a JSON format conversion expert, only return valid JSON format content. Just convert the format, nothing more.",
                        json_schema=STEPS_SCHEMA
                    )
                    return max_resp
                except Exception as e:
                    print(f"chat_max also failed: {e}")
                    return {"step1": "analyze the question", "step2": "review options", "step3": "select answer"}
        except Exception as e:
            if retry < 2:
                continue
            else:
                print(f"doubao failed after 3 retries: {e}")
                return {"step1": "analyze the question", "step2": "review options", "step3": "select answer"}

def ifsolving(question: str, history: str, step: str, context: Any) -> tuple:
    """
    Evaluate if current reasoning steps are sufficient to answer the question.

    This function determines whether the accumulated reasoning provides enough
    information to select a definitive answer from the given options.

    Args:
        question (str): The medical question
        history (str): Previous reasoning steps
        step (str): Current reasoning step
        context (Any): Additional context (currently unused)

    Returns:
        tuple: (answer, decision, reason) where:
            - answer: Selected answer option or "None"
            - decision: "yes" if sufficient, "no" if insufficient
            - reason: Explanation for the decision
    """
    content=f'''
the question is:{question}
the history is:{history}
the current step is:{step}
the context is:{context}
Remember,the ouput JSON format should be:
'''
    content2='''{
  "reason": "Your detailed explanation",
  "decision": "yes" or "no",
  "answer": "an option from 'A/B/C/D/E' or 'None'"
}
Remember,the answer should be an option from A/B/C/D/None'''
    content=content+content2
    chat_client = VLLMChat()

    # Primary VLLM call without retry mechanism
    try:
        judge = chat_client.chat(system_prompt=ifproblemSolving(), user_prompt=content)
    except Exception as e:
        print(f"VLLM call failed: {e}")
        judge = None

    # Use Doubao API to rewrite/correct as proper JSON format
    if judge:
        doubao_prompt = f"""Please convert the following content to correct JSON format:
Original content: {judge}

Please return in the following JSON format:
{{
  "reason": "detailed explanation",
  "decision": "yes" or "no",
  "answer": "A/B/C/D/E or None"
}}

Note: Only return JSON format results, do not include any other text."""
    else:
        doubao_prompt = f"""Please determine if the following information is sufficient to answer the question:
Question: {question}
History: {history}
Current step: {step}
Context: {context}

Please return in the following JSON format:
{{
  "reason": "detailed explanation",
  "decision": "yes" or "no",
  "answer": "A/B/C/D/E or None"
}}

Note: Only return JSON format results, do not include any other text."""

    # Doubao retry mechanism with 3 attempts, adding format guidance after each failure
    for retry in range(3):
        try:
            doubao_resp = chat(user_prompt=doubao_prompt, system_prompt="You are a JSON format conversion expert, only return valid JSON format content. Just convert the format, nothing more.")
            judge_json = json.loads(doubao_resp)
            answer = judge_json["answer"]
            decision = judge_json["decision"]
            reason = judge_json["reason"]
            return answer, decision, reason
        except json.JSONDecodeError as e:
            if retry < 2:
                # Add format guidance to original prompt
                doubao_prompt = (doubao_prompt + "<correct>\n" +
                               "The right format is : {\"reason\": \"explanation\", \"decision\": \"yes\" or \"no\", \"answer\": \"A/B/C/D/E/None\"}\n" +
                               "But last time you output a wrong format:" + "{" + doubao_resp + "}" +
                               "\nNow extract with the right format again.</correct>")
                continue
            else:
                # After 3 failures, use chat_max with structured output
                try:
                    max_resp = chat_max(
                        user_prompt=doubao_prompt,
                        system_prompt="You are a JSON format conversion expert, only return valid JSON format content. Just convert the format, nothing more.",
                        json_schema=JUDGE_SCHEMA
                    )
                    answer = max_resp["answer"]
                    decision = max_resp["decision"]
                    reason = max_resp["reason"]
                    return answer, decision, reason
                except Exception as e:
                    print(f"chat_max also failed: {e}")
                    return "None", "no", "Error in processing"
        except Exception as e:
            if retry < 2:
                continue
            else:
                print(f"doubao failed after 3 retries: {e}")
                return "None", "no", "Error in processing"

def GenStepsNoContent(question: str, history: str, laststep: str) -> Dict[str, str]:
    """
    Generate subsequent reasoning steps without introducing new patient information.

    This function creates next analytical steps based on existing information only,
    following the constraint that no new tests or examinations should be suggested
    in an exam scenario.

    Args:
        question (str): The medical question
        history (str): Previous reasoning steps
        laststep (str): Most recent reasoning step

    Returns:
        Dict[str, str]: Dictionary containing three next reasoning steps
    """
    content = f'''
the question is:{question}
the history is:{history}
the last step is:{laststep}
Remember, all the information about patient have already included in question. So do not suggest new tests, examinations, or patient interactions.
'''
    chat_client = VLLMChat()

    # Primary VLLM call without retry mechanism
    try:
        steps = chat_client.chat(system_prompt=generateStepNoContent(), user_prompt=content)
    except Exception as e:
        print(f"VLLM call failed: {e}")
        steps = None

    # Use Doubao API to rewrite/correct as proper JSON format
    if steps:
        doubao_prompt = f"""Please convert the following content to correct JSON format:
Original content: {steps}

Please return in the following JSON format:
{{
  "step1": "first analytical step",
  "step2": "second analytical step",
  "step3": "third analytical step"
}}

Note: Only return JSON format results, do not include any other text."""
    else:
        doubao_prompt = f"""Please generate three subsequent analytical steps based on the following information:
Question: {question}
History: {history}
Last step: {laststep}

Note: All patient information is already included in the question, do not suggest new tests, examinations, or patient interactions.

Please return in the following JSON format:
{{
  "step1": "first analytical step",
  "step2": "second analytical step",
  "step3": "third analytical step"
}}

Note: Only return JSON format results, do not include any other text."""

    # Doubao retry mechanism with 3 attempts, adding format guidance after each failure
    for retry in range(3):
        try:
            doubao_resp = chat(user_prompt=doubao_prompt, system_prompt="You are a JSON format conversion expert, only return valid JSON format content. Just convert the format, nothing more.")
            steps_json = json.loads(doubao_resp)
            return steps_json
        except json.JSONDecodeError as e:
            if retry < 2:
                # Add format guidance to original prompt
                doubao_prompt = (doubao_prompt + "<correct>\n" +
                               "The right format is : {\"step1\": \"step1\", \"step2\": \"step2\", \"step3\": \"step3\"}\n" +
                               "But last time you output a wrong format:" + "{" + doubao_resp + "}" +
                               "\nNow extract with the right format again.</correct>")
                continue
            else:
                # After 3 failures, use chat_max with structured output
                try:
                    max_resp = chat_max(
                        user_prompt=doubao_prompt,
                        system_prompt="You are a JSON format conversion expert, only return valid JSON format content. Just convert the format, nothing more.",
                        json_schema=STEPS_SCHEMA
                    )
                    return max_resp
                except Exception as e:
                    print(f"chat_max also failed: {e}")
                    return {"step1": "continue analysis", "step2": "evaluate options", "step3": "make decision"}
        except Exception as e:
            if retry < 2:
                continue
            else:
                print(f"doubao failed after 3 retries: {e}")
                return {"step1": "continue analysis", "step2": "evaluate options", "step3": "make decision"}

def ifconvincing(question: str, history: str) -> str:
    """
    Evaluate if the provided reasoning is convincing and sufficient.

    This function assesses the quality, logic, and completeness of the reasoning
    process to determine if it provides a convincing solution to the medical question.

    Args:
        question (str): The medical question
        history (str): The reasoning steps/solution to evaluate

    Returns:
        str: "yes" if reasoning is convincing, "no" otherwise
    """
    content = f"""
the question is:{question}
the solution method is:{history}
"""
    chat_client = VLLMChat()

    # Primary VLLM call without retry mechanism
    try:
        resp = chat_client.chat(system_prompt=ifconvincingprompt(), user_prompt=content)
    except Exception as e:
        print(f"VLLM call failed: {e}")
        resp = None

    # Use Doubao API to rewrite/correct as proper JSON format
    if resp:
        doubao_prompt = f"""Please convert the following content to correct JSON format:
Original content: {resp}

Please return in the following JSON format:
{{
  "reason": "explanation of the reasoning process",
  "convincing?": "yes" or "no"
}}

Note: Only return JSON format results, do not include any other text."""
    else:
        doubao_prompt = f"""Please evaluate the reasoning quality of the following medical question solution:
Question: {question}
Solution: {history}

Please return in the following JSON format:
{{
  "reason": "explanation of the reasoning process",
  "convincing?": "yes" or "no"
}}

Note: Only return JSON format results, do not include any other text."""

    # Doubao retry mechanism with 3 attempts, adding format guidance after each failure
    for retry in range(3):
        try:
            doubao_resp = chat(user_prompt=doubao_prompt, system_prompt="You are a JSON format conversion expert, only return valid JSON format content. Just convert the format, nothing more.")
            resp_json = json.loads(doubao_resp)
            return resp_json['convincing?']
        except json.JSONDecodeError as e:
            if retry < 2:
                # Add format guidance to original prompt
                doubao_prompt = (doubao_prompt + "<correct>\n" +
                               "The right format is : {\"reason\": \"explanation\", \"convincing?\": \"yes\" or \"no\"}\n" +
                               "But last time you output a wrong format:" + "{" + doubao_resp + "}" +
                               "\nNow extract with the right format again.</correct>")
                continue
            else:
                # After 3 failures, use chat_max with structured output
                try:
                    max_resp = chat_max(
                        user_prompt=doubao_prompt,
                        system_prompt="You are a JSON format conversion expert, only return valid JSON format content. Just convert the format, nothing more.",
                        json_schema=CONVINCING_SCHEMA
                    )
                    return max_resp['convincing?']
                except Exception as e:
                    print(f"chat_max also failed: {e}")
                    return "no"
        except Exception as e:
            if retry < 2:
                continue
            else:
                print(f"doubao failed after 3 retries: {e}")
                return "no"

def complete_reasoning_at_max_depth(question: str, history: str, depth: int) -> tuple:
    """
    Complete reasoning process when maximum depth is reached.

    When the Tree-of-Thought reasoning reaches the maximum allowed depth,
    this function asks the VLLM model to complete the remaining reasoning
    process in a single step and provide the final answer.

    Args:
        question (str): The medical question
        history (str): Accumulated reasoning steps
        depth (int): Current reasoning depth

    Returns:
        tuple: (completed_steps, final_answer) where:
            - completed_steps: The final reasoning completion
            - final_answer: The selected answer option
    """
    content = f"""
the question is:{question}
the history is:{history}
the current depth is:{depth}
"""
    chat_client = VLLMChat()

    # Primary VLLM call without retry mechanism
    try:
        completed_reasoning = chat_client.chat(system_prompt=complete_reasoning_prompt(), user_prompt=content)
    except Exception as e:
        print(f"VLLM call failed: {e}")
        completed_reasoning = None

    # Use Doubao API to organize output and extract answer
    if completed_reasoning:
        doubao_prompt = f"""Please extract the final answer and reorganize the following into json format from the following completed reasoning:
Original content: {completed_reasoning}

Please return in the following JSON format:
{{
  "completed_steps": "The completed reasoning steps(the "Continue Reasoning" content)",
  "final_answer": "A/B/C/D/E"
}}

Note: Only return JSON format results, do not include any other text."""
    else:
        # If VLLM call failed, return error information
        return "Error in completion", "OoF"

    # Doubao retry mechanism with 3 attempts, adding format guidance after each failure
    for retry in range(3):
        try:
            doubao_resp = chat(user_prompt=doubao_prompt, system_prompt="You are a JSON format extraction expert, only return valid JSON format content. Just extract the answer, nothing more.")
            resp_json = json.loads(doubao_resp)
            completed_steps = resp_json.get("completed_steps", completed_reasoning)
            final_answer = resp_json.get("final_answer", "OoF")
            return completed_steps, final_answer
        except json.JSONDecodeError as e:
            if retry < 2:
                # Add format guidance to original prompt
                doubao_prompt = (doubao_prompt + "<correct>\n" +
                               "The right format is : {\"completed_steps\": \"reasoning steps\", \"final_answer\": \"A/B/C/D/E\"}\n" +
                               "But last time you output a wrong format:" + "{" + doubao_resp + "}" +
                               "\nNow extract with the right format again.</correct>")
                continue
            else:
                # After 3 failures, use chat_max with structured output
                try:
                    max_resp = chat_max(
                        user_prompt=doubao_prompt,
                        system_prompt="You are a JSON format extraction expert, only return valid JSON format content. Just extract the answer, nothing more.",
                        json_schema={
                            "type": "object",
                            "properties": {
                                "completed_steps": {"type": "string", "description": "The completed reasoning steps(the 'Continue Reasoning' content)"},
                                "final_answer": {"type": "string", "description": "The final answer A/B/C/D/E"}
                            },
                            "required": ["completed_steps", "final_answer"],
                            "additionalProperties": False
                        }
                    )
                    return max_resp["completed_steps"], max_resp["final_answer"]
                except Exception as e:
                    print(f"chat_max also failed: {e}")
                    return completed_reasoning, "OoF"
        except Exception as e:
            if retry < 2:
                continue
            else:
                print(f"doubao failed after 3 retries: {e}")
                return completed_reasoning, "OoF"

# ===================================================================
# Data Loading and Utility Functions
# ===================================================================

def getQA_MedQA(path: str) -> List[Dict[str, Any]]:
    """
    Load medical QA data from JSONL file.

    Args:
        path (str): Path to the JSONL data file

    Returns:
        List[Dict[str, Any]]: List of parsed question data objects
    """
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))  # Parse each line as JSON object
    return data

def write_to_jsonl(result: Dict[str, Any], output_path: str) -> None:
    """
    Write result data to JSONL file in append mode.

    Args:
        result (Dict[str, Any]): Result data to write
        output_path (str): Output file path
    """
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()  # Immediately flush buffer
    except Exception as e:
        print(f"Error writing to file {output_path}: {e}")

def getjsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.

    Args:
        path (str): Path to the JSONL file

    Returns:
        List[Dict[str, Any]]: List of parsed JSON objects
    """
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))  # Parse each line as JSON object
    return data

# ===================================================================
# Concurrent Processing Infrastructure
# ===================================================================

# Global print lock for thread-safe output
print_lock = threading.Lock()

def safe_print(*args, **kwargs) -> None:
    """
    Thread-safe print function with forced output buffer flush.

    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    """
    with print_lock:
        print(*args, **kwargs)
        # Force flush output buffer
        import sys
        sys.stdout.flush()

class OrderedOutputHandler:
    """
    Ensures ordered processing and output of results.

    This handler maintains the order of processing results, writes to files,
    prints to console, and updates statistics. It includes question ID mapping
    and filtering features for mixed evaluation results.
    """

    def __init__(self, total_questions: int, dev_data: List[Dict], output_path: str):
        """
        Initialize the ordered output handler.

        Args:
            total_questions (int): Total number of questions to process
            dev_data (List[Dict]): Development data for question ID matching
            output_path (str): Path for output file
        """
        self.next_expected_index = 0
        self.pending_results = {}
        self.lock = threading.Lock()
        self.total_questions = total_questions
        self.dev_data = dev_data

        # Storage for all records and q_id mapping
        self.all_records = []
        self.q_id_to_records = {}  # q_id -> list of records
        self.mixed_q_ids = set()   # q_ids with both correct and incorrect answers

        # File paths
        self.final_output_path = output_path

        # Built-in statistics counters
        self.right_count = 0
        self.error_count = 0

        self._ensure_directories_exist()

    def _ensure_directories_exist(self) -> None:
        """Ensure output directory exists."""
        import os
        os.makedirs(os.path.dirname(self.final_output_path), exist_ok=True)

    def add_result(self, index: int, result: Dict[str, Any]) -> None:
        """
        Add processing result and process immediately if it's the expected index.

        Args:
            index (int): Result index
            result (Dict[str, Any]): Processing result data
        """
        with self.lock:
            if index == self.next_expected_index:
                self._process_result(result)
                self.next_expected_index += 1
                self._flush_pending_results()
            else:
                self.pending_results[index] = result

    def _add_qid_to_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add question ID to record by matching with development data.

        Args:
            record (Dict[str, Any]): Record to add q_id to

        Returns:
            Dict[str, Any]: Record with added q_id
        """
        question = record["question"]
        for i, dev in enumerate(self.dev_data):
            q = dev["question"]
            option = dev["options"]
            q = str(q) + "\n" "the options are:" + str(option)
            if q == question:
                record["q_id"] = i + 1  # q_id starts from 1
                break
        return record

    def _update_q_id_statistics(self, record: Dict[str, Any]) -> None:
        """
        Update question ID statistics information.

        Args:
            record (Dict[str, Any]): Record to update statistics for
        """
        q_id = record.get("q_id")
        if q_id is None:
            return

        if q_id not in self.q_id_to_records:
            self.q_id_to_records[q_id] = []

        self.q_id_to_records[q_id].append(record)

        # Calculate flag (1 represents correct, -1 represents incorrect)
        if record.get("actual_answer") == record.get("ideal_answer"):
            flag = 1
        else:
            flag = -1

        record["flag"] = flag

    def _process_result(self, result: Dict[str, Any]) -> None:
        """
        Process individual result in order: update statistics, add q_id, check for mixed q_id.

        Args:
            result (Dict[str, Any]): Result to process
        """
        total_processed_before = self.right_count + self.error_count

        # Update statistics
        ans_pair = result.get('ans_pair')
        if result['success'] and ans_pair:
            if ans_pair["actual_answer"] == ans_pair["ideal_answer"]:
                self.right_count += 1
            else:
                self.error_count += 1
        else:
            self.error_count += 1

        total_processed_after = self.right_count + self.error_count

        # Process history_records, add q_id and update statistics
        if result['success'] and 'history_records' in result and result['history_records']:
            for record in result['history_records']:
                record_with_qid = self._add_qid_to_record(record.copy())
                self._update_q_id_statistics(record_with_qid)
                self.all_records.append(record_with_qid)

        # Print summary
        safe_print(f"\n-------------------------------------------------------Current question: question#{result['index']}-------------------------------------------------------")
        if result['success']:
            safe_print(f"\n✅ Question #{result['index']} Processed. ({total_processed_after}/{self.total_questions})")
            if ans_pair:
                raw_answers_str = result['txt_record'].split("Answers: ")[1].split("\n")[0]
                safe_print(f"   ┣━ Ideal Answer:  {ans_pair['ideal_answer']}")
                safe_print(f"   ┣━ Actual Answer: {ans_pair['actual_answer']}")
                safe_print(f"   ┗━ All Generated Answers: {raw_answers_str}")
        else:
            safe_print(f"\n❌ Question #{result['index']} Failed. ({total_processed_after}/{self.total_questions})")
            safe_print(f"   ┗━ Error: {result.get('error', 'Unknown error')}")

        if total_processed_after > 0:
            accuracy = self.right_count / total_processed_after
            safe_print(f"--- 📈 Running Stats: Accuracy = {accuracy:.4f} ({self.right_count}/{total_processed_after}) ---")

    def _flush_pending_results(self) -> None:
        """Process all cached consecutive results."""
        while self.next_expected_index in self.pending_results:
            result = self.pending_results.pop(self.next_expected_index)
            self._process_result(result)
            self.next_expected_index += 1

    def finalize_and_write_filtered_results(self) -> int:
        """
        After all processing is complete, identify mixed q_ids and write final results.

        Mixed q_ids are questions that have both correct and incorrect evaluations,
        indicating variability in the reasoning process.

        Returns:
            int: Number of lines written to output file
        """
        with self.lock:
            safe_print("\nStart identifying mixed q_ids (questions with both correct and incorrect answers)...")

            # Identify mixed q_ids
            for q_id, records in self.q_id_to_records.items():
                flags = set(record.get("flag", 0) for record in records)
                if 1 in flags and -1 in flags:
                    self.mixed_q_ids.add(q_id)

            safe_print(f"Found{len(self.mixed_q_ids)}question IDs with both positive and negative evaluations")
            safe_print(f"Total processed{len(self.q_id_to_records)}different q_ids")

            # Write mixed q_id records to final file
            lines_written = 0
            for record in self.all_records:
                if record.get("q_id") in self.mixed_q_ids:
                    write_to_jsonl(record, self.final_output_path)
                    lines_written += 1

            safe_print(f"Written{lines_written}lines to{self.final_output_path}")
            return lines_written

    def force_flush_all(self) -> None:
        """Force flush all pending results."""
        with self.lock:
            sorted_keys = sorted(self.pending_results.keys())
            for key in sorted_keys:
                if key == self.next_expected_index:
                    self._flush_pending_results()

class QuestionProcessor:
    """
    Individual question processor that encapsulates the complete processing logic.

    This class handles the processing of a single medical question through the
    Tree-of-Thought reasoning pipeline.
    """

    def __init__(self, question_data: Dict[str, Any], index: int):
        """
        Initialize question processor.

        Args:
            question_data (Dict[str, Any]): Question data containing question, options, and answer
            index (int): Question index in the dataset
        """
        self.index = index
        self.question_data = question_data
        self.question = question_data["question"]
        self.option = question_data["options"]
        self.full_question = str(self.question) + "\n" "the options are:" + str(self.option)
        self.ideal_answer = question_data["answer_idx"]

    def process(self) -> Dict[str, Any]:
        """
        Process a single question through the complete pipeline.

        Returns:
            Dict[str, Any]: Processing result containing answers and history
        """
        try:
            step1s = self._generate_first_step()
            answers, history_records = process_steps(
                self.full_question,
                "the step chain just begin:",
                step1s,
                ideal_answer=self.ideal_answer
            )
            processed_output = self._process_answers(answers)

            final_result = {
                'index': self.index,
                'success': True,
                'ans_pair': processed_output.get('ans_pair'),
                'txt_record': processed_output.get('txt_record'),
                'history_records': history_records
            }
            return final_result

        except Exception as e:
            safe_print(f"Error processing question #{self.index}: {e}")
            return {
                'index': self.index,
                'success': False,
                'error': str(e)
            }

    def _generate_first_step(self) -> Dict[str, str]:
        """
        Generate initial reasoning steps for the question.

        Returns:
            Dict[str, str]: Three initial reasoning steps
        """
        while True:
            try:
                step1s = firststepGenToT(self.full_question)
                break
            except Exception as e:
                safe_print(f"An error occurred in Step1, QuestionID=#{self.index}")
        return step1s

    def _process_answers(self, answers: List[str]) -> Dict[str, Any]:
        """
        Process answers and return formatted result.

        Args:
            answers (List[str]): List of generated answer options

        Returns:
            Dict[str, Any]: Processed answer pair and text record
        """
        if not answers:
            answers = ["OoF"]

        counter = Counter(answers)
        most_common_list = counter.most_common(2)
        most_common1 = "OoF"  # default value
        if most_common_list:
            if most_common_list[0][0] != 'OoF':
                most_common1 = most_common_list[0][0]
            elif len(most_common_list) > 1:
                most_common1 = most_common_list[1][0]

        ans_pair = {"ideal_answer": self.ideal_answer, "actual_answer": most_common1}
        txt_record = f'''Index: {self.index}\nQuestion: {self.full_question}\nAnswers: {answers}\n{ans_pair}\n--------------------------------------------------------------------------------------------------------------------------------------------\n'''

        return {
            'ans_pair': ans_pair,
            'txt_record': txt_record
        }

# ===================================================================
# Main Processing Functions
# ===================================================================

def process_steps(
    question: str,
    history: str,
    steps: Dict[str, str],
    ideal_answer: str,
    depth: int = 1,
    max_depth: int = 2
) -> tuple:
    """
    Process reasoning steps using Tree-of-Thought methodology.

    This function implements the core ToT reasoning logic:
    1. For each step, evaluate if current information is sufficient
    2. If sufficient, record the answer and reasoning
    3. If insufficient and not at max depth, generate next steps recursively
    4. If at max depth, complete reasoning in one final step

    Args:
        question (str): Medical question to answer
        history (str): Previous reasoning steps
        steps (Dict[str, str]): Current reasoning steps to evaluate
        ideal_answer (str): Correct answer from dataset
        depth (int): Current reasoning depth
        max_depth (int): Maximum allowed reasoning depth

    Returns:
        tuple: (results, history_records) where:
            - results: List of generated answers
            - history_records: List of detailed reasoning records
    """
    results = []
    history_records = []

    for key, step in steps.items():
        new_history = f"{history}\nStep{depth} is: {step}."

        # Process if solving evaluation
        actual_answer, decision, reason = ifsolving(question, new_history, step, context=None)

        if decision == "yes":
            results.append(actual_answer)
            record_history = new_history + "\nReason:" + reason + "\nSo the answer is " + actual_answer
            dic = {
                "question": question,
                "solution": record_history,
                "ideal_answer": ideal_answer,
                "actual_answer": actual_answer,
                "deepmax": False  # Reached answer before maximum depth
            }
            history_records.append(dic)

        elif decision == "no":
            if depth < max_depth:
                # Process next step generation
                step2s = GenStepsNoContent(question=question, history=new_history, laststep=step)
                sub_results, sub_history_records = process_steps(
                    question, new_history, step2s, ideal_answer, depth + 1
                )
                results.extend(sub_results)
                history_records.extend(sub_history_records)
            else:
                # Reached maximum depth, let VLLM model complete remaining reasoning
                print("history:")
                print(new_history)
                print(f"Reached maximum depth({depth}), now letting VLLM complete remaining reasoning...")

                completed_reasoning, completed_answer = complete_reasoning_at_max_depth(
                    question, new_history, depth
                )
                results.append(completed_answer)

                # Build complete record history including completion reasoning
                complete_record_history = new_history + "\n[Remaining Steps]: " + completed_reasoning + "\nSo the answer is " + completed_answer
                dic = {
                    "question": question,
                    "solution": complete_record_history,
                    "ideal_answer": ideal_answer,
                    "actual_answer": completed_answer,
                    "deepmax": True  # Marked as reasoning done at maximum depth
                }
                history_records.append(dic)

    return results, history_records

# Global variables for signal handling
global_writer = None
shutdown_requested = False

def signal_handler(signum: int, frame) -> None:
    """
    Signal handler function to ensure graceful program exit.

    Args:
        signum (int): Signal number received
        frame: Current stack frame
    """
    global shutdown_requested, global_writer
    safe_print(f"\nReceived signal {signum}, Gracefully exiting...")
    shutdown_requested = True

    if global_writer:
        safe_print("Saving all unwritten data...")
        global_writer.force_flush_all()
        safe_print("Data saving completed")

    sys.exit(0)

def process_questions_concurrently(
    qa_data: List[Dict[str, Any]],
    dev_data: List[Dict[str, Any]],
    output_path: str,
    max_workers: int = 40
) -> tuple:
    """
    Process questions concurrently using ThreadPoolExecutor.

    This function manages the concurrent processing of multiple medical questions
    while maintaining ordered output and handling graceful shutdowns.

    Args:
        qa_data (List[Dict[str, Any]]): Test questions to process
        dev_data (List[Dict[str, Any]]): Development data for q_id matching
        output_path (str): Output file path
        max_workers (int): Maximum number of concurrent worker threads

    Returns:
        tuple: (all_results, right_count, error_count, lines_written)
    """
    global global_writer, shutdown_requested

    all_results = {}

    # Use new handler
    handler = OrderedOutputHandler(
        total_questions=len(qa_data),
        dev_data=dev_data,
        output_path=output_path
    )
    global_writer = handler

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}
        for i, question_data in enumerate(qa_data):
            processor = QuestionProcessor(question_data, i)
            future = executor.submit(processor.process)
            future_to_index[future] = i

        with tqdm(total=len(qa_data), desc="Processing questions progress") as pbar:
            for future in concurrent.futures.as_completed(future_to_index):
                if shutdown_requested:
                    safe_print("Shutdown signal detected, stop processing new tasks...")
                    break

                index = future_to_index[future]
                try:
                    result = future.result()

                    if result['success']:
                        handler.add_result(index, result)
                        ans_pair = result.get('ans_pair')
                        if ans_pair:
                            all_results[index] = ans_pair
                    else:
                        error_result = {
                            'index': index,
                            'success': False,
                            'error': result.get('error', 'Unknown error')
                        }
                        handler.add_result(index, error_result)

                except Exception as e:
                    safe_print(f"❗️ A future for question #{index} failed with an exception: {e}")
                    error_result = {
                        'index': index,
                        'success': False,
                        'error': str(e)
                    }
                    handler.add_result(index, error_result)

                pbar.update(1)

    handler.force_flush_all()

    # Final processing: identify mixed q_ids and write filtered results
    lines_written = handler.finalize_and_write_filtered_results()

    return all_results, handler.right_count, handler.error_count, lines_written

# ===================================================================
# Main Execution Function
# ===================================================================

def main_with_config(config) -> Dict[str, Any]:
    """
    Run Step 1 using configuration object.

    This function orchestrates the complete medical question processing pipeline
    using the provided configuration settings.

    Args:
        config: Configuration object containing file paths and settings

    Returns:
        Dict[str, Any]: Processing statistics and results
    """
    # Configuration paths
    test_path = config.test_data_path
    dev_path = config.dev_data_path
    output_path = config.step1_output_path

    # Read data
    qa_data = getQA_MedQA(test_path)
    dev_data = getQA_MedQA(dev_path)

    safe_print(f"Start processing{len(qa_data)}test questions...")
    safe_print(f"use{len(dev_data)}development data for q_id matching...")

    # Execute concurrent processing
    totRAGqa, right1, error1, lines_written = process_questions_concurrently(
        qa_data, dev_data, output_path, max_workers=config.step1_max_workers
    )

    # Output final results
    safe_print(f"\n🎉 Step1 Completion Statistics:")
    safe_print(f"Correct count: {right1}")
    safe_print(f"Error count: {error1}")
    safe_print(f"Total processed count: {right1 + error1}")
    safe_print(f"Number of filtered records written: {lines_written}")

    if right1 + error1 > 0:
        accuracy = float(right1) / float(right1 + error1)
        safe_print(f"Final accuracy: {accuracy:.4f}")

    return {
        'total_processed': right1 + error1,
        'correct_count': right1,
        'error_count': error1,
        'lines_written': lines_written,
        'accuracy': accuracy if right1 + error1 > 0 else 0.0
    }

if __name__ == "__main__":
    # Check if configuration file exists, use it if available, otherwise use default paths
    try:
        from config import config
        print("✅ Run using configuration file Step1")
        results = main_with_config(config)
    except ImportError:
        print("⚠️  Configuration file not found, use default configuration to run Step1")

        # Default configuration
        class DefaultConfig:
            def __init__(self):
                self.test_data_path = os.environ.get("TEST_DATA_PATH", "/path/to/test/data.jsonl")
                self.dev_data_path = os.environ.get("DEV_DATA_PATH", "/path/to/dev/data.jsonl")
                self.step1_output_path = os.environ.get("STEP1_OUTPUT_PATH", "/path/to/output/data.jsonl")
                self.step1_max_workers = 20

        config = DefaultConfig()
        results = main_with_config(config)