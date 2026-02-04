# Med-REFL: Enhancing Complex Reasoning via Fine-grained Self-Correction


## ⚡ Introduction

**Med-REFL**  is a novel framework designed to enhance the complex reasoning capabilities of Large Language Models (LLMs) in the medical domain.

Diverging from traditional methods, Med-REFL focuses on improving the model's **internal reflection process**. It leverages the Tree-of-Thought (ToT) paradigm to explore diverse reasoning pathways and automatically constructs a high-quality Direct Preference Optimization (DPO) dataset. This approach trains the model to identify flaws in its own reasoning and perform self-correction, thereby boosting accuracy and reliability on complex medical problems without the need for expensive expert annotation.


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


