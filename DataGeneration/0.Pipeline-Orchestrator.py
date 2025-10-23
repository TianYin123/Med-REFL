#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical AI Pipeline - Main Orchestrator Module.
Coordinates execution of the three-step pipeline for medical question processing
and DPO (Direct Preference Optimization) training data generation.

Pipeline Steps:
1. QM-ToT-VLLM-ToTRollout-Merged.py: Medical question processing with Tree-of-Thought reasoning
2. Tree-Reconstruction-Fix.py: Solution tree reconstruction and DPO pair generation
3. DPOPair-Construction.py: Final DPO text construction and formatting
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from config import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️  config.py not found, using default configuration")

try:
    import importlib.util

    def load_step_module(step_name: str, filename: str):
        """
        Dynamically loads a pipeline step module from file.

        Args:
            step_name: Name identifier for the module
            filename: Path to the module file

        Returns:
            Loaded Python module

        Raises:
            FileNotFoundError: If the module file doesn't exist
        """
        module_path = Path(__file__).parent / filename
        if not module_path.exists():
            raise FileNotFoundError(f"Step file does not exist: {module_path}")

        spec = importlib.util.spec_from_file_location(step_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # Load all three pipeline step modules
    step1_module = load_step_module("step1", "1. QM-ToT-VLLM-ToTRollout-Merged.py")
    step2_module = load_step_module("step2", "2.Tree-Reconstruction-Fix.py")
    step3_module = load_step_module("step3", "3.DPOPair-Construction.py")

except Exception as e:
    print(f"❌ Module loading failed: {e}")
    print("Please ensure all three step files are in the current directory")
    sys.exit(1)


def setup_logging(log_dir: str = "logs"):
    """
    Configures the logging system for pipeline execution.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


class PipelineExecutor:
    """
    Main pipeline execution coordinator.
    Manages the sequential execution of all pipeline steps with error handling,
    progress tracking, and comprehensive reporting.
    """

    def __init__(self, config_obj=None):
        """
        Initialize the pipeline executor.

        Args:
            config_obj: Configuration object (optional, defaults to None)
        """
        self.config = config_obj
        self.logger = setup_logging()
        self.results = {}
        self.start_time = None

    def print_banner(self):
        """
        Displays the pipeline execution banner with configuration details.
        Shows pipeline overview and current settings.
        """
        print("=" * 100)
        print("🏥 Medical AI Pipeline - Complete Pipeline")
        print("=" * 100)
        print("📋 Execution Steps:")
        print("   Step 1: QM-ToT-VLLM-ToTRollout-Merged.py (Medical Question ToT Reasoning)")
        print("   Step 2: Tree-Reconstruction-Fix.py (Tree Reconstruction and DPO Pair Generation)")
        print("   Step 3: DPOPair-Construction.py (DPO Text Construction)")
        print("=" * 100)

        if self.config:
            print("⚙️  Configuration:")
            self.config.print_config()
        else:
            print("⚙️  Using default configuration")

        print("=" * 100)
        print()

    def validate_prerequisites(self) -> bool:
        """
        Validates that all prerequisites for pipeline execution are met.
        Checks input files, directories, and system requirements.

        Returns:
            True if all prerequisites are satisfied, False otherwise
        """
        self.logger.info("🔍 Validating pipeline execution prerequisites...")

        # Check input files if configuration is available
        if self.config:
            validation_results = self.config.validate_paths()

            # Check required input files
            if not validation_results.get('test_data', False):
                self.logger.error(f"❌ Test data file does not exist: {self.config.test_data_path}")
                return False
            if not validation_results.get('dev_data', False):
                self.logger.error(f"❌ Development data file does not exist: {self.config.dev_data_path}")
                return False

            # Create output directories
            self.config.ensure_output_dirs()

        self.logger.info("✅ Prerequisites validation completed successfully")
        return True

    def execute_step(self, step_name: str, step_func, step_description: str) -> Dict[str, Any]:
        """
        Executes a single pipeline step with error handling and timing.

        Args:
            step_name: Identifier for the step
            step_func: Function to execute for this step
            step_description: Human-readable description of the step

        Returns:
            Dictionary containing execution results or error information
        """
        self.logger.info(f"🚀 Starting execution of {step_name}: {step_description}")
        step_start_time = time.time()

        try:
            # Execute the step function
            result = step_func()

            step_duration = time.time() - step_start_time
            self.logger.info(f"✅ {step_name} execution completed, duration: {step_duration:.2f} seconds")

            # Log key result metrics
            if isinstance(result, dict):
                self.logger.info(f"📊 {step_name} result statistics:")
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"   - {key}: {value}")
                    elif isinstance(value, str) and len(value) < 100:
                        self.logger.info(f"   - {key}: {value}")

            return result

        except Exception as e:
            step_duration = time.time() - step_start_time
            self.logger.error(f"❌ {step_name} execution failed, duration: {step_duration:.2f} seconds")
            self.logger.error(f"Error details: {e}", exc_info=True)
            return {"error": str(e), "duration": step_duration}

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Executes the complete three-step pipeline sequentially.
        Manages dependencies between steps and provides comprehensive error handling.

        Returns:
            Dictionary containing results from all pipeline steps
        """
        self.start_time = time.time()
        self.print_banner()

        # Validate prerequisites
        if not self.validate_prerequisites():
            return {"error": "Prerequisites validation failed"}

        self.logger.info("🎯 Starting complete pipeline execution...")

        # Step 1: Tree-of-Thought reasoning
        if self.config and hasattr(step1_module, 'main_with_config'):
            step1_result = self.execute_step(
                "Step 1",
                lambda: step1_module.main_with_config(self.config),
                "QM-ToT-VLLM-ToTRollout-Merged"
            )
        else:
            step1_result = self.execute_step(
                "Step 1",
                step1_module.main,
                "QM-ToT-VLLM-ToTRollout-Merged (Default configuration)"
            )

        self.results['step1'] = step1_result

        # Check if Step 1 succeeded
        if 'error' in step1_result:
            self.logger.error("❌ Step 1 execution failed, terminating pipeline")
            return self.results

        # Step 2: Tree Reconstruction and DPO Pair Generation
        if self.config and hasattr(step2_module, 'main_with_config'):
            step2_result = self.execute_step(
                "Step 2",
                lambda: step2_module.main_with_config(self.config),
                "Tree-Reconstruction-Fix"
            )
        else:
            step2_result = self.execute_step(
                "Step 2",
                step2_module.main,
                "Tree-Reconstruction-Fix (Default configuration)"
            )

        self.results['step2'] = step2_result

        # Check if Step 2 succeeded
        if 'error' in step2_result:
            self.logger.error("❌ Step 2 execution failed, terminating pipeline")
            return self.results

        # Step 3: DPO Text Construction
        if self.config and hasattr(step3_module, 'main_with_config'):
            step3_result = self.execute_step(
                "Step 3",
                lambda: step3_module.main_with_config(self.config),
                "DPOPair-Construction"
            )
        else:
            step3_result = self.execute_step(
                "Step 3",
                step3_module.main,
                "DPOPair-Construction (Default configuration)"
            )

        self.results['step3'] = step3_result

        # Generate final execution report
        self.generate_final_report()

        return self.results

    def generate_final_report(self):
        """
        Generates and displays a comprehensive execution report.
        Shows statistics, timing, and file information for all completed steps.
        """
        total_duration = time.time() - self.start_time

        print("\n" + "=" * 100)
        print("📊 Medical AI Pipeline Execution Report")
        print("=" * 100)

        # Report results for each step
        for step_name, result in self.results.items():
            print(f"\n🔹 {step_name.upper()}:")
            if 'error' in result:
                print(f"   ❌ Status: Execution failed")
                print(f"   📝 Error: {result['error']}")
            else:
                print(f"   ✅ Status: Execution successful")
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key != 'error':
                            if isinstance(value, float):
                                print(f"   📈 {key}: {value:.4f}")
                            elif isinstance(value, int):
                                print(f"   🔢 {key}: {value}")
                            else:
                                print(f"   📄 {key}: {str(value)[:100]}...")

        print(f"\n⏱️  Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")

        # Check and report on output files
        if self.config:
            final_output = self.config.step3_output_path
            if os.path.exists(final_output):
                file_size = os.path.getsize(final_output) / (1024 * 1024)  # MB
                print(f"📁 Final output file: {final_output}")
                print(f"📏 File size: {file_size:.2f} MB")

                # Count data rows
                with open(final_output, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                print(f"📝 Number of data rows: {line_count}")
            else:
                print(f"⚠️  Final output file does not exist: {final_output}")

        print("=" * 100)

        # Save execution report to file
        self.save_execution_report(total_duration)

    def save_execution_report(self, total_duration: float):
        """
        Saves detailed execution report to a JSON file.

        Args:
            total_duration: Total execution time in seconds
        """
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)

        report_filename = f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(report_dir, report_filename)

        report_data = {
            "pipeline_name": "Medical AI Pipeline",
            "execution_time": datetime.now().isoformat(),
            "total_duration_seconds": total_duration,
            "total_duration_minutes": total_duration / 60,
            "config_used": CONFIG_AVAILABLE,
            "steps_results": self.results,
            "success": all('error' not in result for result in self.results.values())
        }

        if self.config:
            report_data["config"] = {
                "test_data_path": self.config.test_data_path,
                "dev_data_path": self.config.dev_data_path,
                "step1_output_path": self.config.step1_output_path,
                "step2_output_path": self.config.step2_output_path,
                "step3_output_path": self.config.step3_output_path,
                "N_CHOSEN": self.config.N_CHOSEN,
                "M_REJECTED": self.config.M_REJECTED,
                "dpo_format": self.config.dpo_format
            }

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"📋 Execution report saved to: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save execution report: {e}")


def main():
    """
    Main entry point for pipeline execution.
    Creates executor, runs pipeline, and handles exit codes.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        # Create pipeline executor with available configuration
        executor = PipelineExecutor(config if CONFIG_AVAILABLE else None)

        # Execute complete pipeline
        results = executor.run_full_pipeline()

        # Check overall execution status
        if all('error' not in result for result in results.values()):
            print("\n🎉 Pipeline execution completed successfully!")
            return 0
        else:
            print("\n⚠️  Pipeline execution partially failed, please check the above logs")
            return 1

    except KeyboardInterrupt:
        print("\n\n⏹️  User interrupted execution")
        return 2
    except Exception as e:
        print(f"\n\n💥 Unexpected error occurred during pipeline execution: {e}")
        logging.error(f"Pipeline execution error: {e}", exc_info=True)
        return 3


if __name__ == "__main__":
    """
    Direct execution entry point.
    Handles command line execution and system exit.
    """
    exit_code = main()
    sys.exit(exit_code)