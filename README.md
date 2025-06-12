# Med-REFL: Medical Reasoning Enhancement via Self-Corrected Fine-grained Reflection

<p align="center">
    📃 <a href="https://arxiv.org/abs/2504.12334" target="_blank">Paper</a> │ 📚 <a href="https://huggingface.co/datasets/HANI-LAB/Med-REFL-DPO" target="_blank">Dataset</a> │ 🤗 <a href="https://huggingface.co/HANI-LAB/Med-REFL-Llama-3.1-8B-lora" target="_blank">LoRA Weights</a> 
</p>

## 🚀 Updates
* **[June 12, 2025]** We have open-sourced the Med-REFL DPO dataset, all corresponding LoRA weights, and the evaluation code.
* **[Upcoming]** The data generation scripts are currently being refactored and are expected to be released in September 2025.

## ⚡ Introduction

**Med-REFL** (Medical Reasoning Enhancement via self-corrected Fine-grained refLection) is a novel framework designed to enhance the complex reasoning capabilities of Large Language Models (LLMs) in the medical domain.

Diverging from traditional methods, Med-REFL focuses on improving the model's **internal reflection process**. It leverages the Tree-of-Thought (ToT) paradigm to explore diverse reasoning pathways and automatically constructs a high-quality Direct Preference Optimization (DPO) dataset. This approach trains the model to identify flaws in its own reasoning and perform self-correction, thereby boosting accuracy and reliability on complex medical problems without the need for expensive expert annotation.

<div align=center>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/189_distil-whisper/thumbnail.jpg"  width="90%" alt="Med-REFL Concept" align=center/>
</div>
<p align="center">
<em>The Med-REFL framework enhances model reasoning through self-corrected, fine-grained reflection.</em>
</p>

## 🧩 Assets
We have open-sourced all LoRA weights and the DPO dataset generated using the Med-REFL framework.

### LoRA Weights
Apply our LoRA weights to the following base models to significantly improve their medical reasoning performance.

| LoRA for Base Model        | Backbone     | Link                                                         |
| :------------------------- | :----------- | :------------------------------------------------------------------------ |
| **Med-REFL for Llama-3.1-8B** | Llama-3.1-8B | [🤗](https://huggingface.co/HANI-LAB/Med-REFL-Llama-3.1-8B-lora)     |
| **Med-REFL for Qwen2.5-7B** | Qwen2.5-7B   | [🤗](https://huggingface.co/HANI-LAB/Med-REFL-Qwen2.5-7B-lora)       |
| **Med-REFL for Huatuo-o1-8B** | Huatuo-o1-8B | [🤗](https://huggingface.co/HANI-LAB/Med-REFL-Huatuo-o1-8B-lora)     |
| **Med-REFL for MedReason-8B** | MedReason-8B | [🤗](https://huggingface.co/HANI-LAB/Med-REFL-MedReason-8B-lora)     |

### DPO Dataset
Our DPO dataset is hosted on the Hugging Face Hub and includes the following components:
<table>
    <thead>
        <tr>
            <th>Data Component</th>
            <th>Sub-Component</th>
            <th>Description</th>
            <th>Link</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2"><strong>Med-REFL</strong></td>
            <td>Reasoning Enhancement</td>
            <td>Contains ~12,000 preference pairs contrasting correct reasoning paths with plausible but incorrect ones to improve general reasoning discernment.</td>
            <td rowspan="4" align="center"><a href="https://huggingface.co/datasets/HANI-LAB/Med-REFL-DPO" target="_blank">🤗</a></td>
        </tr>
        <tr>
            <td>Reflection Enhancement</td>
            <td>Contains ~21,000 preference pairs designed to train error detection and self-correction by distinguishing effective reflections from flawed ones.</td>
        </tr>
        <tr>
            <td rowspan="2"><strong>Ablation Study Data</strong></td>
            <td>Huatuo-o1 Random CoT</td>
            <td>Random rollout Chain-of-Thought pairs used in ablation studies to benchmark against the main Med-REFL data, generated for the Huatuo-o1 model.</td>
        </tr>
        <tr>
            <td>Llama3.1-8b Random CoT</td>
            <td>Random rollout Chain-of-Thought pairs used in ablation studies to benchmark against the main Med-REFL data, generated for the Llama3.1-8b model.</td>
        </tr>
    </tbody>
</table>
<h2>📊 Performance</h2>
<p>
    Extensive experiments demonstrate that Med-REFL significantly enhances the medical reasoning capabilities of various large language models. On the primary MedQA-USMLE benchmark, our methodology yields a substantial average accuracy improvement of <strong>+4.11%</strong> across four baseline models. It not only instills sophisticated reasoning skills in general-purpose models like Llama3.1-8B (+5.82%) but also further augments models already specialized for medical reasoning, such as Huatuo-o1 (+4.13%).
</p>
<p>
    Furthermore, Med-REFL exhibits strong generalization, boosting performance across a diverse suite of unseen medical question-answering datasets and proving its ability to foster robust and transferable reasoning skills.
</p>

<h3>MedQA-USMLE Benchmark Results</h3>
<p>The following table shows the performance improvements on the MedQA-USMLE test set after applying Med-REFL fine-tuning.</p>
<table border="1" cellpadding="5" cellspacing="0">
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Original</th>
            <th><strong>Med-REFL</strong></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2"><strong>Instruction-Tuned</strong></td>
            <td>Qwen2.5-7B</td>
            <td>57.11</td>
            <td><strong>59.70</strong> <span style="color: #2E8B57; font-size: small;">(+2.59)</span></td>
        </tr>
        <tr>
            <td>Llama3.1-8B</td>
            <td>59.92</td>
            <td><strong>65.74</strong> <span style="color: #2E8B57; font-size: small;">(+5.82)</span></td>
        </tr>
        <tr>
            <td rowspan="2"><strong>Medical Reasoning</strong></td>
            <td>Huatuo-o1</td>
            <td>69.59</td>
            <td><strong>73.72</strong> <span style="color: #2E8B57; font-size: small;">(+4.13)</span></td>
        </tr>
        <tr>
            <td>MedReason-8B</td>
            <td>66.27</td>
            <td><strong>70.16</strong> <span style="color: #2E8B57; font-size: small;">(+3.89)</span></td>
        </tr>
        <tr>
            <td colspan="2"><strong>Average (%)</strong></td>
            <td>63.22</td>
            <td><strong>67.33</strong> <span style="color: #2E8B57; font-size: small;">(+4.11)</span></td>
        </tr>
    </tbody>
</table>

<h3>Generalization Ability on Various Benchmarks(Out of Distribution)</h3>
<p>Med-REFL's effectiveness extends to other challenging medical benchmarks, demonstrating robust generalization.</p>
<table border="1" cellpadding="5" cellspacing="0">
    <thead>
        <tr>
            <th>Benchmark</th>
            <th>Model</th>
            <th>Original</th>
            <th><strong>+ MedREFL</strong></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3"><strong>MedMCQA</strong></td>
            <td>Huatuo-o1</td>
            <td>62.13</td>
            <td><strong>64.66</strong> <span style="color: #2E8B57; font-size: small;">(+2.53)</span></td>
        </tr>
        <tr>
            <td>Llama3.1-8B</td>
            <td>57.61</td>
            <td><strong>59.11</strong> <span style="color: #2E8B57; font-size: small;">(+1.50)</span></td>
        </tr>
        <tr>
            <td>MedReason</td>
            <td>58.98</td>
            <td><strong>59.78</strong> <span style="color: #2E8B57; font-size: small;">(+0.80)</span></td>
        </tr>
        <tr>
            <td rowspan="3"><strong>GPQA (Med+)</strong></td>
            <td>Huatuo-o1</td>
            <td>50.67</td>
            <td><strong>56.80</strong> <span style="color: #2E8B57; font-size: small;">(+6.13)</span></td>
        </tr>
        <tr>
            <td>Llama3.1-8B</td>
            <td>45.16</td>
            <td><strong>50.22</strong> <span style="color: #2E8B57; font-size: small;">(+5.06)</span></td>
        </tr>
        <tr>
            <td>MedReason</td>
            <td>45.64</td>
            <td><strong>49.84</strong> <span style="color: #2E8B57; font-size: small;">(+4.20)</span></td>
        </tr>
        <tr>
            <td rowspan="3"><strong>MMLU-Pro (Med+)</strong></td>
            <td>Huatuo-o1</td>
            <td>61.87</td>
            <td><strong>64.97</strong> <span style="color: #2E8B57; font-size: small;">(+3.10)</span></td>
        </tr>
        <tr>
            <td>Llama3.1-8B</td>
            <td>57.56</td>
            <td><strong>60.89</strong> <span style="color: #2E8B57; font-size: small;">(+3.30)</span></td>
        </tr>
        <tr>
            <td>MedReason</td>
            <td>59.14</td>
            <td><strong>62.51</strong> <span style="color: #2E8B57; font-size: small;">(+3.37)</span></td>
        </tr>
    </tbody>
</table>

## 🛠️ Training
We use [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) for model training. Follow the steps below to reproduce our training setup.

#### 1. Prepare the Data
Our DPO dataset consists of two main parts. First, merge them into a single file for training using the provided script.
```bash
python train/merge_data.py
```

#### 2. Set Environment Variables
Configure the environment variables according to your machine's setup.
```bash
export FORCE_TORCHRUN=1
export CUDA_VISIBLE_DEVICES=0,1
```


#### 3. Start Training
Use the `llamafactory-cli` and our provided configuration file to start training.
```bash
llamafactory-cli train --config train/train_config.yaml
```
All training parameters, such as model paths, dataset paths, and hyperparameters, are predefined in the `train/train_config.yaml` file.

## 🧐 Evaluation
The evaluation process consists of two steps: generating model outputs and then verifying the results.

You can find sample outputs in the `evaluate/results/` folder for reference and to facilitate reproduction of our paper's results.

#### 1. Generate Model Outputs

Run the `evaluate-generate.py` script to have the model generate answers for the questions in the test set (located in `evaluate/data/`).

#### 2. Verify Results

After generating the answers, run the `evaluate-verification.py` script to automatically score the outputs and calculate the accuracy.


## 📖 Citation
If you use our code, data, or weights in your research, please consider citing our paper:
```

```
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TianYin123/Med-REFL&type=Date)](https://star-history.com/#TianYin123/Med-REFL&Date)
