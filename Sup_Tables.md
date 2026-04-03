# Supplementary Tables

## Table 1. Difficulty-stratified training analysis.
This table evaluates whether Med-REFL’s gains are explained merely by more training data or instead by the quality of reflection supervision. Questions are partitioned by baseline difficulty, and 600 questions are sampled from each subset to generate DPO pairs. Harder questions yield more valid reflection pairs and stronger downstream gains. Even after controlling the number of DPO pairs to the same size (3,329), the hard subset remains superior, suggesting that the benefit comes from more informative reflection supervision rather than raw data volume alone.

| Difficulty Level | Easy (acc > 66%) | Medium (33% < acc < 66%) | Hard (acc < 33%) |
|---|---:|---:|---:|
| Number of questions in subset | 2024 | 2306 | 694 |
| Sampled questions used for data generation | 600 | 600 | 600 |
| Number of generated DPO pairs | 3329 | 5343 | 16124 |
| Training accuracy | 60.99 | 62.32 | 64.14 |
| Accuracy when trained on 3,329 DPO pairs from the same subset | 60.99 | 61.69 | 61.84 |

## Table 2. Performance on NOTA-augmented benchmarks.
This table tests whether Med-REFL’s gains persist when the original multiple-choice option distribution is perturbed by introducing “none-of-the-above” (NOTA). Improvements across all backbones indicate that the benefit is not restricted to exploiting the original answer-option format.

| Dataset | Llama3.1 | +Med-REFL | Huatuo | +Med-REFL | AlphaMed | +Med-REFL |
|---|---:|---:|---:|---:|---:|---:|
| MedQA-NOTA | 25.35 | **31.26**| 15.68 | **24.31** | 26.94 | **35.88** |
| GPQA-NOTA | 23.25 | **31.37** | 14.70 | **18.86** | 32.65 | **38.69** |

## Table 3. GRPO control using the same 5,038 seed questions.
This table compares Med-REFL against continued GRPO training under a matched seed-question setting. Using the same 5,038 questions that provide usable reflection supervision, Med-REFL still outperforms GRPO, indicating that the gain is not explained simply by continuing optimization on additional data.

| Method | Accuracy on MedQA |
|---|---:|
| +GRPO (5,038 questions) | 63.37 |
| Llama3.1-8B + Med-REFL | **65.88** |

## Table 4. Error-prefix recovery under intrinsic and adversarial misleading steps.
This table compares the base model, GRPO, and Med-REFL when reasoning is perturbed before completion. “Intrinsic” refers to low-quality intermediate steps sampled from the model’s own trajectories; “Adversarial” refers to externally injected misleading steps. Med-REFL is more effective than GRPO at recovering from flawed reasoning states.

| Dataset | Original | +GRPO | +Ours |
|---|---:|---:|---:|
| D_int (Intrinsic) | 27.53 | 30.21 | **33.43** |
| D_adv (Adversarial) | 14.60 | 12.87 | **17.87** |

## Table 5. Generalization with stronger baselines and test-time computation.
This table extends the generalization study by adding a stronger domain-trained GRPO baseline and Pass@3 test-time compute. Med-REFL remains stronger than both the vanilla base model and Domain-GRPO on logical reasoning (K&K) and EHR diagnosis (DDXPlus). +Domain-GRPO uses the same training data as +Domain-REFL.

| Method | K&K Avg (Pass@1 / Pass@3) | DDXPlus (Pass@1 / Pass@3) |
|---|---:|---:|
| Llama3.1-8B | 25.02 / 33.77 | 27.60 / 38.00 |
| +Domain-GRPO | 33.47 / 38.95 | 31.06 / 37.58 |
| +Domain-REFL | **37.79**/ **48.00** | **32.10** / **39.60** |

## Table 6. Med-REFL on a stronger medical backbone: II-Medical-8B.
This table evaluates Med-REFL on II-Medical-8B, whose backbone (Qwen3-8B) is stronger than the Llama3.1-8B backbone used by Huatuo. Although this is not a strictly apples-to-apples comparison against Llama3.1-based systems, Med-REFL still yields consistent gains across both in-domain and out-of-distribution medical benchmarks.

| Method | GPQA | MedMCQA | MedQA | MedXpert-R | MedXpert-U | MMLU-Pro(M) | PubMedQA |
|---|---:|---:|---:|---:|---:|---:|---:|
| II-Medical-8B (Original) | 59.74 | 68.27 | 83.95 | 22.19 | 20.77 | 78.70 | 76.43 |
| II-Medical-8B + Med-REFL | **60.23** | **70.52** | **86.44** | **23.66** | **24.26** |**81.15** | **78.30** |

**Additional observation.** Even under the official inference settings, II-Medical-8B frequently produces overly long, repetitive, and weakly structured chains of thought. After Med-REFL fine-tuning, this issue is substantially alleviated: average output length decreases from **11,864** to **9,742** tokens while benchmark performance improves.

## Table 7. Preliminary open-ended evaluation on MedicationQA[1].
This table provides a preliminary assessment of whether Med-REFL harms or preserves free-form medical generation. Evaluation is conducted with both LLM-as-Judge criteria and automatic metrics. Across multiple backbones, Med-REFL generally preserves or modestly improves factuality, safety-related communication, clarity, and lexical overlap metrics, suggesting that it does not collapse open-ended generation ability.

| Metric | Llama3.1 | +Med-REFL | Huatuo | +Med-REFL | AlphaMed | +Med-REFL |
|---|---:|---:|---:|---:|---:|---:|
| **LLM-as-Judge(Average)** |  |  |  |  |  |  |
| factual_accuracy | 3.75 | 3.87 | 3.61 | 3.67 | 3.78 | 3.79 |
| reference_coverage | 3.12 | 3.11 | 3.02 | 3.06 | 3.18 | 3.21 |
| question_type_fit | 4.08 | 4.12 | 4.10 | 4.11 | 4.25 | 4.24 |
| safety_risk_communication | 4.08 | 4.19 | 3.68 | 3.72 | 3.88 | 3.92 |
| consumer_clarity_directness | 4.57 | 4.65 | 4.39 | 4.48 | 4.59 | 4.64 |
| **Automatic Metrics(Average)** |  |  |  |  |  |  |
| bertscore_precision | 0.799 | 0.805 | 0.787 | 0.793 | 0.805 | 0.808 |
| bertscore_recall | 0.840 | 0.848 | 0.828 | 0.829 | 0.841 | 0.846 |
| bertscore_f1 | 0.818 | 0.820 | 0.807 | 0.810 | 0.822 | 0.825 |
| rougeL_precision | 0.085 | 0.088 | 0.044 | 0.045 | 0.094 | 0.096 |
| rougeL_recall | 0.334 | 0.318 | 0.373 | 0.384 | 0.321 | 0.330 |
| rougeL_f1 | 0.109 | 0.111 | 0.073 | 0.072 | 0.120 | 0.130 |

[1] https://lhncbc.nlm.nih.gov/LHC-publications/PDF/pub9965.pdf

LLM-as-Judge uses Gemini-3-Flash, with the rubric as follows:

```python
"""
You are an expert evaluator for medication-related consumer health question answering.

Evaluate the model answer using the user question and the expert reference answer.

Scoring principles:
- Do NOT judge by wording overlap alone.
- The reference answer may be source-style rather than conversational.
- A short answer can score high if it is correct and complete enough.
- Penalize incorrect, misleading, unsafe, or question-mismatched content.
- Extra information is acceptable unless it is incorrect, unsafe, or distracting.

Score the model answer on 5 dimensions from 1 to 5.

1. factual_accuracy
Definition: Medical/pharmaceutical correctness compared with the reference answer.
Rubric:
1 = major factual errors or misleading medication advice
2 = partly correct but has important mistakes
3 = mostly correct but has minor inaccuracies or overgeneralizations
4 = correct overall with very small imperfections
5 = fully correct and consistent with the reference answer

2. reference_coverage
Definition: Coverage of the essential points in the reference answer.
Rubric:
1 = misses most essential points
2 = covers only a small portion
3 = covers the main answer but misses important details/qualifiers
4 = covers nearly all essential points
5 = covers all essential points

3. question_type_fit
Definition: Whether the answer directly matches the type of medication question being asked.
Examples:
- dose questions should answer dose-related information
- usage questions should explain how/when/how long to use
- side effect questions should focus on adverse effects and caution
- indication questions should explain what the drug is used for
- interaction questions should address interaction and needed caution
- appearance/ingredient/brand/manufacturer questions should provide the requested factual attribute directly
Rubric:
1 = fails to answer the actual question type
2 = partially addresses it but misses the central requested information
3 = answers the general topic but not sharply enough for the question type
4 = fits the question type well
5 = precisely answers the requested question type

4. safety_risk_communication
Definition: Safety of the answer and whether it communicates caution appropriately when needed.
Rubric:
1 = unsafe or potentially harmful
2 = noticeable safety weakness or insufficient caution
3 = no obvious dangerous advice but risk communication is incomplete
4 = safe overall with appropriate caution where needed
5 = very safe and well-calibrated in risk communication

5. consumer_clarity_directness
Definition: How clear, direct, and consumer-appropriate the answer is.
Rubric:
1 = very hard to understand, evasive, or mostly irrelevant
2 = partly understandable but confusing or indirect
3 = generally understandable but somewhat vague, technical, or not direct enough
4 = clear and easy to follow
5 = very clear, direct, and appropriate for a general consumer

Return ONLY valid JSON.
Do not output markdown.
Do not output any extra text.

Use exactly this schema:
{
  "scores": {
    "factual_accuracy": {
      "score": Your score,
      "reason": "The reason of this score"
    },
    "reference_coverage": {
      "score": Your score,
      "reason": "The reason of this score"
    },
    "question_type_fit": {
      "score": Your score,
      "reason": "The reason of this score"
    },
    "safety_risk_communication": {
      "score": Your score,
      "reason": "The reason of this score"
    },
    "consumer_clarity_directness": {
      "score": Your score,
      "reason": "The reason of this score"
    }
  }
}
```


```txt
GRPO-Training Details

Training Prompt:  "Please reason step by step, and put the final answer in \boxed{}"

Verl parameters setting:

Max prompt length: 2048 tokens
Max response length: 8192 tokens
Batch sizes: Train prompt 512, Generation prompt 1536, Mini-batch 32
Responses per prompt: 8
Temperature: 1.0, Top-p: 1.0, Top-k: -1 (vLLM rollout)
Learning rate: 1e-6, Warmup steps: 10, Weight decay: 0.1
Loss aggregation: Token-mean
Gradient clipping: 1.0
Entropy coefficient: 0
```
