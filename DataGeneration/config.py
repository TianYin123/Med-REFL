#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management module for the Medical AI Pipeline.
Centralizes all configuration parameters including file paths, API settings,
and processing parameters for the three-step pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any


class PipelineConfig:
    """
    Central configuration class for the Medical AI Pipeline.
    Manages file paths, API configurations, and processing parameters
    for all three pipeline steps.
    """

    def __init__(self):
        # Base directory configuration
        self.root_dir = Path(__file__).parent

        # Input data file paths
        self.test_data_path = os.environ.get("TEST_DATA_PATH", "/path/to/test/data.jsonl")
        self.dev_data_path = os.environ.get("DEV_DATA_PATH", "/path/to/dev/data.jsonl")

        # Output file paths for each pipeline step
        self.step1_output_path = os.environ.get("STEP1_OUTPUT_PATH", "/path/to/step1/output.jsonl")
        self.step2_output_dir = os.environ.get("STEP2_OUTPUT_DIR", "/path/to/step2/output")
        self.step2_output_path = os.path.join(self.step2_output_dir, 'dpo_reflection_pairs.jsonl')
        self.step3_output_path = os.path.join(self.step2_output_dir, 'final_dpo_pairs_with_text.jsonl')

        # API configuration for Doubao service (used in Step 1)
        self.doubao_api_key = os.environ.get("ARK_API_KEY", "your-api-key-here")

        # API configuration for OpenRouter service (backup API for Step 1)
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "your-openrouter-key-here")

        # Local VLLM service configuration (used in Steps 2 and 3)
        self.vllm_api_base = "http://localhost:8001/v1"
        self.vllm_model_path = os.environ.get("VLLM_MODEL_PATH", "/path/to/your/model")
        self.vllm_api_key = "EMPTY"  # Local VLLM typically doesn't require API key

        # DPO algorithm parameters for Step 2
        self.N_CHOSEN = 2  # Number of top alternative solutions to select as "chosen"
        self.M_REJECTED = 2  # Number of alternative solutions to select as "rejected"
        self.K_POST_REASONING = 6  # Number of candidate solutions for post-reasoning analysis
        self.K_CANDIDATES_FOR_LLM = 4  # Number of error candidates to present to LLM

        # DPO text generation parameters for Step 3
        self.dpo_format = "tags"  # Output format: "tags" or "think_conclusion"
        self.dpo_max_workers = 10  # Number of concurrent worker threads

        # Performance tuning parameters
        self.step1_max_workers = 20  # Concurrent workers for Step 1 processing

        # Logging and debugging configuration
        self.log_level = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR
        self.enable_progress_bars = True  # Whether to display progress bars

        # LLM generation parameters
        self.temperature = 0.7  # Randomness in LLM generation (0.0-1.0)
        self.max_tokens = 4096  # Maximum token count for LLM responses

        # Retry and timeout configuration
        self.max_retries = 3  # Maximum number of retry attempts for API calls
        self.retry_delay = 1  # Delay between retry attempts in seconds
        self.api_timeout = 300  # API call timeout in seconds
        self.step_timeout = 3600  # Single step timeout in seconds

    def ensure_output_dirs(self):
        """
        Creates all necessary output directories if they don't exist.
        Ensures the pipeline has proper directory structure for writing results.
        """
        os.makedirs(self.step2_output_dir, exist_ok=True)

    def validate_paths(self) -> Dict[str, bool]:
        """
        Validates that all input files exist and are accessible.
        Returns a dictionary indicating which files are present.
        """
        validation_results = {}

        # Check input data files
        validation_results['test_data'] = os.path.exists(self.test_data_path)
        validation_results['dev_data'] = os.path.exists(self.dev_data_path)

        # Check intermediate output files (for incremental processing)
        validation_results['step1_output'] = os.path.exists(self.step1_output_path)
        validation_results['step2_output'] = os.path.exists(self.step2_output_path)

        return validation_results

    def get_step1_args(self) -> Dict[str, Any]:
        """
        Returns configuration parameters specifically for Step 1.
        Includes paths and processing settings for ToT reasoning.
        """
        return {
            'test_path': self.test_data_path,
            'dev_path': self.dev_data_path,
            'output_path': self.step1_output_path,
            'max_workers': self.step1_max_workers
        }

    def get_step2_args(self) -> Dict[str, Any]:
        """
        Returns configuration parameters specifically for Step 2.
        Includes DPO parameters and paths for tree reconstruction.
        """
        return {
            'input_file': self.step1_output_path,
            'output_dir': self.step2_output_dir,
            'N_CHOSEN': self.N_CHOSEN,
            'M_REJECTED': self.M_REJECTED,
            'K_POST_REASONING': self.K_POST_REASONING,
            'K_CANDIDATES_FOR_LLM': self.K_CANDIDATES_FOR_LLM
        }

    def get_step3_args(self) -> Dict[str, Any]:
        """
        Returns configuration parameters specifically for Step 3.
        Includes formatting options and processing settings for final text generation.
        """
        return {
            'input_path': self.step2_output_path,
            'output_path': self.step3_output_path,
            'format': self.dpo_format,
            'max_workers': self.dpo_max_workers
        }

    def print_config(self):
        """
        Displays the current configuration in a formatted manner.
        Shows all file paths, API settings, and processing parameters.
        """
        print("=" * 80)
        print("🔧 Medical AI Pipeline Configuration")
        print("=" * 80)
        print(f"📁 Root Directory: {self.root_dir}")
        print(f"📄 Test Data: {self.test_data_path}")
        print(f"📄 Dev Data: {self.dev_data_path}")
        print(f"📤 Step1 Output: {self.step1_output_path}")
        print(f"📤 Step2 Output: {self.step2_output_path}")
        print(f"📤 Step3 Output: {self.step3_output_path}")
        print(f"⚙️  Step1 Max Workers: {self.step1_max_workers}")
        print(f"⚙️  Step2 DPO Settings: N={self.N_CHOSEN}, M={self.M_REJECTED}")
        print(f"⚙️  Step3 Format: {self.dpo_format}, Workers: {self.dpo_max_workers}")
        print("=" * 80)

    def check_environment(self) -> Dict[str, bool]:
        """
        Performs comprehensive environment and configuration validation.
        Checks file existence, API configuration, and directory permissions.
        """
        checks = {}

        # Check input data files
        checks['test_data_exists'] = os.path.exists(self.test_data_path)
        checks['dev_data_exists'] = os.path.exists(self.dev_data_path)

        # Check API key configuration
        checks['doubao_api_configured'] = bool(
            self.doubao_api_key and self.doubao_api_key != "your-api-key-here"
        )
        checks['openrouter_api_configured'] = bool(
            self.openrouter_api_key and self.openrouter_api_key != "your-openrouter-key-here"
        )

        # Check VLLM model configuration
        checks['vllm_model_exists'] = (
            os.path.exists(self.vllm_model_path)
            if self.vllm_model_path != "/path/to/your/model"
            else False
        )

        # Check output directory write permissions
        try:
            os.makedirs(self.step2_output_dir, exist_ok=True)
            checks['output_dir_writable'] = True
        except Exception:
            checks['output_dir_writable'] = False

        return checks

    def print_environment_check(self):
        """
        Displays detailed environment check results.
        Shows the status of all configuration elements and provides guidance.
        """
        checks = self.check_environment()

        print("🔍 Environment Check Results:")
        print("=" * 50)

        # Data file validation
        if checks['test_data_exists']:
            print("✅ Test data file exists")
        else:
            print(f"❌ Test data file does not exist: {self.test_data_path}")

        if checks['dev_data_exists']:
            print("✅ Development data file exists")
        else:
            print(f"❌ Development data file does not exist: {self.dev_data_path}")

        # API configuration validation
        if checks['doubao_api_configured']:
            print("✅ Doubao API configured")
        else:
            print("⚠️  Doubao API not configured or using default values")

        if checks['openrouter_api_configured']:
            print("✅ OpenRouter API configured")
        else:
            print("⚠️  OpenRouter API not configured or using default values")

        # VLLM configuration validation
        if checks['vllm_model_exists']:
            print("✅ VLLM model file exists")
        else:
            print(f"⚠️  VLLM model file does not exist: {self.vllm_model_path}")
            print("   Please ensure VLLM service is running and model path is correct")

        # Output directory validation
        if checks['output_dir_writable']:
            print("✅ Output directory is writable")
        else:
            print("❌ Output directory is not writable")

        print("=" * 50)

        # Overall validation summary
        all_good = all([
            checks['test_data_exists'],
            checks['dev_data_exists'],
            checks['output_dir_writable']
        ])

        if all_good:
            print("🎉 Basic configuration check passed! Ready to start pipeline execution.")
        else:
            print("⚠️  Please fix the above issues and run again.")


# Global configuration instance for easy access
config = PipelineConfig()


if __name__ == "__main__":
    """
    Configuration module entry point.
    Displays current configuration and performs environment validation.
    """
    print("🔧 Medical AI Pipeline Configuration")
    print("=" * 60)
    config.print_config()
    print()
    config.print_environment_check()