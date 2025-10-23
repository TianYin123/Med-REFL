# Medical AI Pipeline with Tree-of-Thought Reasoning and DPO Training

A comprehensive medical AI pipeline for processing clinical exam questions using Tree-of-Thought (ToT) reasoning methodology and generating Direct Preference Optimization (DPO) training data for medical question answering systems.

## 🎯 Overview

This pipeline implements a three-step process for transforming medical examination questions into high-quality training data:

1. **Question Processing with Tree-of-Thought Reasoning** - Multi-step analytical reasoning for medical questions
2. **Solution Tree Reconstruction & DPO Pair Generation** - Error analysis and preference pair creation
3. **DPO Text Construction** - Final training text generation with multiple output formats

## 🏗️ Architecture

### Pipeline Components

```
Medical Questions → ToT Reasoning → Solution Trees → DPO Pairs → Training Text
```

#### Step 1: Question Processing (`1. QM-ToT-VLLM-ToTRollout-Merged.py`)
- Implements Tree-of-Thought reasoning for medical question analysis
- Integrates multiple LLM services (Doubao, OpenRouter, local VLLM)
- Uses concurrent processing for scalability
- Generates step-by-step analytical solutions with evaluation logic

#### Step 2: Tree Reconstruction (`2.Tree-Reconstruction-Fix.py`)
- Reconstructs enhanced solution trees from reasoning paths
- Implements LLM-based error location using medical expertise
- Generates DPO reflection pairs through tree analysis
- Supports both mid-reasoning and post-reasoning error types

#### Step 3: Text Construction (`3.DPOPair-Construction.py`)
- Converts structured DPO pairs into natural language training data
- Supports multiple output formats (XML tags vs Think-Conclusion)
- Applies text augmentation for training diversity
- Uses concurrent processing for efficient generation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Access to LLM APIs (Doubao, OpenRouter, or local VLLM)
- Medical question dataset in JSONL format

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd medical-ai-pipeline
```

2. Install dependencies:
```bash
pip install openai requests tqdm volcenginesdkarkruntime
```

3. Set up environment variables:
```bash
export ARK_API_KEY="your-api-key-here"
export OPENROUTER_API_KEY="your-openrouter-key-here"
export TEST_DATA_PATH="/path/to/test/data.jsonl"
export DEV_DATA_PATH="/path/to/dev/data.jsonl"
export VLLM_MODEL_PATH="/path/to/your/model"
```

### Configuration

Edit `config.py` to customize:
- File paths for input/output data
- API keys and endpoints
- Processing parameters (worker counts, DPO settings)
- LLM model configurations

### Running the Pipeline

#### Option 1: Complete Pipeline Execution
```bash
python "0.Pipeline-Orchestrator.py"
```

#### Option 2: Step-by-Step Execution
```bash
# Step 1: Tree-of-Thought Reasoning
python "1. QM-ToT-VLLM-ToTRollout-Merged.py"

# Step 2: Tree Reconstruction and DPO Generation
python "2.Tree-Reconstruction-Fix.py"

# Step 3: Final Text Construction
python "3.DPOPair-Construction.py"
```

## 📁 Project Structure

```
├── config.py                          # Configuration management
├── 0.Pipeline-Orchestrator.py         # Main pipeline coordinator
├── 1. QM-ToT-VLLM-ToTRollout-Merged.py # Step 1: ToT reasoning
├── 2.Tree-Reconstruction-Fix.py       # Step 2: Tree reconstruction
├── 3.DPOPair-Construction.py          # Step 3: Text generation
├── data/                              # Input data directory
├── output/                            # Output directory
├── logs/                              # Log files
└── reports/                           # Execution reports
```

## ⚙️ Configuration

### Key Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `N_CHOSEN` | Number of top correct solutions for DPO | 2 |
| `M_REJECTED` | Number of top incorrect solutions for DPO | 2 |
| `K_POST_REASONING` | Candidates for post-reasoning analysis | 6 |
| `K_CANDIDATES_FOR_LLM` | Error candidates for LLM evaluation | 4 |
| `dpo_format` | Output format (`tags` or `think_conclusion`) | `tags` |
| `step1_max_workers` | Concurrent workers for Step 1 | 20 |
| `dpo_max_workers` | Concurrent workers for Step 3 | 10 |

### API Configuration

The pipeline supports multiple LLM backends:

1. **Doubao API** (Primary)
   - Used for main reasoning tasks
   - Configure with `ARK_API_KEY`

2. **OpenRouter API** (Backup)
   - Used for structured output with JSON schema validation
   - Configure with `OPENROUTER_API_KEY`

3. **Local VLLM** (Optional)
   - Used for Steps 2 and 3
   - Configure with `VLLM_API_BASE` and `VLLM_MODEL_PATH`

## 📊 Output Formats

### XML Tags Format
```xml
<thinking>
[Initial reasoning process]
</thinking>

<reflection>
[Error analysis and correction]
</reflection>

<conclusion>
[Final answer and summary]
</conclusion>
```

### Think-Conclusion Format
```markdown
## Thinking
[Initial reasoning process]

[Reflection indicator]
[Error analysis and correction]

## Conclusion
[Final answer and summary]
```

## 🔧 Advanced Usage

### Custom LLM Integration

To integrate a custom LLM service:

1. Modify the `VLLMChat` class in the respective step files
2. Update API endpoints and authentication
3. Adjust prompt templates if needed

### Batch Processing

For large datasets, adjust worker counts:
```python
# In config.py
self.step1_max_workers = 40  # Increase for Step 1
self.dpo_max_workers = 20     # Increase for Step 3
```

### Output Format Selection

Choose output format based on your training requirements:
```python
# XML tags format (default)
self.dpo_format = "tags"

# Think-Conclusion format
self.dpo_format = "think_conclusion"
```

## 📈 Performance Metrics

The pipeline provides detailed statistics including:
- Question processing accuracy
- DPO pair generation counts
- Mid-reasoning vs post-reasoning pair distribution
- Processing time and throughput
- Error rates and fallback usage

## 🛠️ Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Reduce worker counts in configuration
   - Implement retry mechanisms (built-in)

2. **Memory Usage**
   - Process data in smaller batches
   - Monitor system resources during execution

3. **Model Availability**
   - Ensure VLLM service is running for local models
   - Verify API keys and endpoint accessibility

### Logging

Enable detailed logging:
```python
# In config.py
self.log_level = "DEBUG"
```

Log files are saved in the `logs/` directory with timestamps.

## 📚 Data Format

### Input Data Format

Medical questions should be in JSONL format:
```json
{"question": "Medical question text", "options": "A. Option 1\nB. Option 2\n...", "answer_idx": "A", "q_id": 1}
```

### Output Data Format

Final DPO training data:
```json
{
  "q_id": 1,
  "question": "Complete question with options",
  "type": "mid-reasoning",
  "Spub": "Public reasoning steps",
  "serr": "Error step",
  "Snew_chosen": "Correct reasoning path",
  "Snew_rejected": "Alternative reasoning path",
  "r_chosen": "Generated training text (chosen)",
  "r_rejected": "Generated training text (rejected)"
}
```

## 🔬 Research Applications

This pipeline is designed for:
- **Medical Education**: Creating teaching examples with error analysis
- **AI Training**: Generating preference data for medical LLMs
- **Clinical Reasoning Research**: Analyzing decision-making processes
- **Assessment Development**: Creating varied question solutions

## 📄 License

This project is part of academic research in medical AI and natural language processing. Please refer to the accompanying license file for usage terms.

## 🤝 Contributing

Contributions are welcome for:
- Performance optimizations
- Additional LLM integrations
- New output formats
- Bug fixes and improvements

## 📞 Support

For technical support or questions:
- Check the log files in `logs/` directory
- Review the configuration documentation
- Verify API accessibility and credentials

---

**Note**: This pipeline is designed for research and educational purposes in medical AI. Ensure compliance with relevant healthcare data regulations and ethical guidelines when using patient data.