# Med-REFL: Enhancing Complex Reasoning via Fine-grained Self-Correction


## ⚡ Introduction

**Med-REFL** (Medical Reasoning Enhancement via self-corrected Fine-grained refLection) is a novel framework designed to enhance the complex reasoning capabilities of Large Language Models (LLMs) in the medical domain.

Diverging from traditional methods, Med-REFL focuses on improving the model's **internal reflection process**. It leverages the Tree-of-Thought (ToT) paradigm to explore diverse reasoning pathways and automatically constructs a high-quality Direct Preference Optimization (DPO) dataset. This approach trains the model to identify flaws in its own reasoning and perform self-correction, thereby boosting accuracy and reliability on complex medical problems without the need for expensive expert annotation.


## 📊 Performance
Extensive experiments demonstrate that Med-REFL consistently and significantly enhances the medical reasoning capabilities across a diverse suite of large language models. On the primary **MedQA-USMLE** benchmark, our methodology yields a substantial average accuracy improvement of **+3.67%** across seven baseline modelsThe framework proves highly versatile, instilling sophisticated reasoning in general-purpose models like Llama3.1-8B (+5.82%), further augmenting models already specialized for medical reasoning such as Huatuo-o1 (+4.13%), and even refining reason-heavy models like Deepseek-Distill-8B (+6.15%).

Furthermore, Med-REFL exhibits strong generalization, boosting performance across a diverse suite of unseen medical question-answering datasets. Its impact is particularly pronounced on benchmarks demanding deep deductive reasoning, with average gains of **+3.53%** on GPQA (Med+) and **+2.20%** on MMLU-Pro (Med+). This proves Med-REFL's ability to foster robust and transferable reasoning skills, a conclusion validated by consistent gains on the complex MedXpert diagnostic benchmark.

### MedQA-USMLE Benchmark Results
The following table shows the performance improvements on the MedQA-USMLE test set after applying Med-REFL fine-tuning, with models grouped by their primary training focus.
<table border="1" cellpadding="5" cellspacing="0">
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Original</th>
            <th><strong>+ Med-REFL</strong></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2"><strong>Instruction-Tuned</strong></td>
            <td>Llama3.1-8B</td>
            <td>59.92</td>
            <td><strong>65.74</strong> <span style="color: #2E8B57; font-size: small;">(+5.82)</span></td>
        </tr>
        <tr>
            <td>Qwen2.5-7B</td>
            <td>57.11</td>
            <td><strong>59.70</strong> <span style="color: #2E8B57; font-size: small;">(+2.59)</span></td>
        </tr>
        <tr>
            <td rowspan="2"><strong>Reason-Heavy</strong></td>
            <td>Huatuo-o1-8B</td>
            <td>69.59</td>
            <td><strong>73.72</strong> <span style="color: #2E8B57; font-size: small;">(+4.13)</span></td>
        </tr>
        <tr>
            <td>Deepseek-Distill-8b</td>
            <td>48.85</td>
            <td><strong>55.00</strong> <span style="color: #2E8B57; font-size: small;">(+6.15)</span></td>
        </tr>
        <tr>
            <td rowspan="2"><strong>Knowledge-Heavy</strong></td>
            <td>MedReason-8B</td>
            <td>66.27</td>
            <td><strong>70.16</strong> <span style="color: #2E8B57; font-size: small;">(+3.89)</span></td>
        </tr>
        <tr>
            <td>UltraMedical3.1-8b</td>
            <td>71.34</td>
            <td><strong>73.08</strong> <span style="color: #2E8B57; font-size: small;">(+1.74)</span></td>
        </tr>
        <tr>
            <td><strong>Pure-RL (GRPO)</strong></td>
            <td>AlphaMed-8b</td>
            <td>65.79</td>
            <td><strong>67.17</strong> <span style="color: #2E8B57; font-size: small;">(+1.38)</span></td>
        </tr>
        <tr>
            <td colspan="2"><strong>Average (%)</strong></td>
            <td>62.11</td>
            <td><strong>65.78</strong> <span style="color: #2E8B57; font-size: small;">(+3.67)</span></td>
        </tr>
    </tbody>
</table>

### Generalization Ability on Various Benchmarks (Out of Distribution)
Med-REFL's effectiveness extends to other challenging medical benchmarks, demonstrating robust generalization across a range of models and tasks.
<table border="1" cellpadding="5" cellspacing="0">
    <thead>
        <tr>
            <th>Benchmark</th>
            <th>Model</th>
            <th>Original</th>
            <th><strong>+ Med-REFL</strong></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="7"><strong>MedMCQA</strong></td>
            <td>Llama3.1-8B</td>
            <td>57.61</td>
            <td><strong>59.11</strong> <span style="color: #2E8B57; font-size: small;">(+1.50)</span></td>
        </tr>
        <tr>
            <td>Qwen2.5-7b</td>
            <td>54.52</td>
            <td><strong>55.79</strong> <span style="color: #2E8B57; font-size: small;">(+1.27)</span></td>
        </tr>
        <tr>
            <td>Huatuo-o1-8b</td>
            <td>62.13</td>
            <td><strong>64.66</strong> <span style="color: #2E8B57; font-size: small;">(+2.53)</span></td>
        </tr>
        <tr>
            <td>Deepseek-Distill-8b</td>
            <td>49.51</td>
            <td><strong>50.69</strong> <span style="color: #2E8B57; font-size: small;">(+1.18)</span></td>
        </tr>
        <tr>
            <td>MedReason-8b</td>
            <td>58.98</td>
            <td><strong>59.78</strong> <span style="color: #2E8B57; font-size: small;">(+0.80)</span></td>
        </tr>
        <tr>
            <td>UltraMedical3.1-8b</td>
            <td>63.30</td>
            <td><strong>64.31</strong> <span style="color: #2E8B57; font-size: small;">(+1.01)</span></td>
        </tr>
        <tr>
            <td>AlphaMed-8b</td>
            <td>61.22</td>
            <td><strong>61.54</strong> <span style="color: #2E8B57; font-size: small;">(+0.32)</span></td>
        </tr>
        <tr>
            <td rowspan="6"><strong>GPQA (Med+)</strong></td>
            <td>Llama3.1-8B</td>
            <td>45.16</td>
            <td><strong>50.22</strong> <span style="color: #2E8B57; font-size: small;">(+5.06)</span></td>
        </tr>
        <tr>
            <td>Qwen2.5-7b</td>
            <td>43.34</td>
            <td><strong>45.36</strong> <span style="color: #2E8B57; font-size: small;">(+2.02)</span></td>
        </tr>
        <tr>
            <td>Huatuo-o1-8b</td>
            <td>50.67</td>
            <td><strong>56.80</strong> <span style="color: #2E8B57; font-size: small;">(+6.13)</span></td>
        </tr>
        <tr>
            <td>Deepseek-Distill-8b</td>
            <td>53.40</td>
            <td><strong>56.01</strong> <span style="color: #2E8B57; font-size: small;">(+2.61)</span></td>
        </tr>
        <tr>
            <td>MedReason-8b</td>
            <td>45.64</td>
            <td><strong>49.84</strong> <span style="color: #2E8B57; font-size: small;">(+4.20)</span></td>
        </tr>
        <tr>
            <td>UltraMedical3.1-8b</td>
            <td>62.43</td>
            <td><strong>64.70</strong> <span style="color: #2E8B57; font-size: small;">(+2.27)</span></td>
        </tr>
        <tr>
            <td rowspan="7"><strong>MMLU-Pro (Med+)</strong></td>
            <td>Llama3.1-8B</td>
            <td>57.56</td>
            <td><strong>60.89</strong> <span style="color: #2E8B57; font-size: small;">(+3.33)</span></td>
        </tr>
        <tr>
            <td>Qwen2.5-7b</td>
            <td>53.83</td>
            <td><strong>55.30</strong> <span style="color: #2E8B57; font-size: small;">(+1.47)</span></td>
        </tr>
        <tr>
            <td>Huatuo-o1-8b</td>
            <td>61.87</td>
            <td><strong>64.97</strong> <span style="color: #2E8B57; font-size: small;">(+3.10)</span></td>
        </tr>
        <tr>
            <td>Deepseek-Distill-8b</td>
            <td>63.06</td>
            <td><strong>63.43</strong> <span style="color: #2E8B57; font-size: small;">(+0.37)</span></td>
        </tr>
        <tr>
            <td>MedReason-8b</td>
            <td>59.14</td>
            <td><strong>62.51</strong> <span style="color: #2E8B57; font-size: small;">(+3.37)</span></td>
        </tr>
        <tr>
            <td>UltraMedical3.1-8b</td>
            <td>63.29</td>
            <td><strong>64.81</strong> <span style="color: #2E8B57; font-size: small;">(+1.52)</span></td>
        </tr>
        <tr>
            <td>AlphaMed-8b</td>
            <td>69.30</td>
            <td><strong>71.50</strong> <span style="color: #2E8B57; font-size: small;">(+2.20)</span></td>
        </tr>
        <tr>
            <td rowspan="6"><strong>PubMedQA</strong></td>
            <td>Llama3.1-8B</td>
            <td>76.26</td>
            <td><strong>77.11</strong> <span style="color: #2E8B57; font-size: small;">(+0.85)</span></td>
        </tr>
        <tr>
            <td>Qwen2.5-7b</td>
            <td>74.12</td>
            <td><strong>73.79</strong> <span style="color: #CD5C5C; font-size: small;">(-0.33)</span></td>
        </tr>
        <tr>
            <td>Huatuo-o1-8b</td>
            <td>77.17</td>
            <td><strong>78.06</strong> <span style="color: #2E8B57; font-size: small;">(+0.89)</span></td>
        </tr>
        <tr>
            <td>Deepseek-Distill-8b</td>
            <td>78.03</td>
            <td><strong>78.17</strong> <span style="color: #2E8B57; font-size: small;">(+0.14)</span></td>
        </tr>
        <tr>
            <td>MedReason-8b</td>
            <td>78.60</td>
            <td><strong>79.12</strong> <span style="color: #2E8B57; font-size: small;">(+0.52)</span></td>
        </tr>
        <tr>
            <td>UltraMedical3.1-8b</td>
            <td>79.02</td>
            <td><strong>80.32</strong> <span style="color: #2E8B57; font-size: small;">(+1.30)</span></td>
        </tr>
    </tbody>
</table>

## 🛠️ Training
We use [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) for model training. Follow the steps below to reproduce our training setup.

#### 0. Download Data
Our Data in ”Data“ folder.
```
python A_merge_data.py
```

#### 1. Prepare the Data
Our DPO dataset consists of two main parts. First, merge them into a single file for training using the provided script.
```bash
python train/merge_data.py
```

#### 2. Set Environment Variables
Configure the environment variables according to your machine's setup.
```bash
export FORCE_TORCHRUN=2
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


