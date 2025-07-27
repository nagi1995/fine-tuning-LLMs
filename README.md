# 📝 Fine-Tuning FLAN-T5 on SAMSum for Dialogue Summarization

This repository explores fine-tuning the [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) model on the [`knkarthick/samsum`](https://huggingface.co/datasets/knkarthick/samsum) dataset for abstractive dialogue summarization. It compares **baseline**, **full fine-tuning**, and **parameter-efficient fine-tuning (LoRA)** approaches using ROUGE and BLEU evaluation metrics.

---

## 📚 Dataset

I used the [samsum dataset](https://huggingface.co/datasets/knkarthick/samsum), which contains human-written summaries of dialogues. 

---

## 🧠 Model

- **Base model:** [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base)
- **Tokenization:** Applied custom prompts during tokenization:  
```python
summarize the below conversation
{{dialogue}}

summary:
```

---

## 🔧 Fine-Tuning Approaches

| Method             | Description |
|--------------------|-------------|
| **Baseline**        | Evaluated pre-trained `flan-t5-base` without fine-tuning |
| **Full Fine-Tuning**| Fine-tuned all model parameters for 4 epochs |
| **LoRA Fine-Tuning**| Trained only low-rank adapters using PEFT LoRA for 5 epochs |

### LoRA Configuration

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
````

---

## 📉 Training Loss Summary

### 🔁 Full Fine-Tuning (4 Epochs)

| Epoch | Train Loss | Val Loss |
| ----- | ---------- | -------- |
| 1     | 1.6931     | 1.5183   |
| 2     | 1.2985     | 1.5067   |
| 3     | 1.0097     | 1.5259   |
| 4     | 0.7771     | 1.6407   |

* **Training Loss Trend:** There's a steady decrease in training loss across epochs, indicating the model is learning effectively from the training data.
* **Validation Loss Behavior:**

  * The validation loss *initially improves* slightly from 1.5183 to 1.5067, but then **starts increasing** in epochs 3 and 4.
  * This suggests **overfitting** — the model continues to fit the training data better while generalization to unseen data deteriorates.
* **Conclusion:** Although full fine-tuning led to the lowest training loss (0.7771), the increasing validation loss indicates that the model may have begun to memorize rather than generalize.

### 🧩 LoRA Fine-Tuning (5 Epochs)

| Epoch | Train Loss | Val Loss |
| ----- | ---------- | -------- |
| 1     | 1.4631     | 1.3962   |
| 2     | 1.4308     | 1.3912   |
| 3     | 1.4035     | 1.3825   |
| 4     | 1.3822     | 1.3823   |
| 5     | 1.3674     | 1.3800   |

* **Training Loss Trend:** LoRA training loss decreases gradually over all 5 epochs, indicating consistent improvement.
* **Validation Loss Behavior:**

  * Validation loss steadily decreases and **stabilizes** by epoch 4–5.
  * The gap between training and validation loss is **much narrower** than in full fine-tuning.
* **Conclusion:** LoRA demonstrates **better generalization** with stable validation loss and no clear signs of overfitting. Even though training loss is higher than full fine-tuning, the validation performance is competitive — and even slightly better in ROUGE/BLEU.

---

## 📊 Evaluation Results (on SAMSum test set)

| Metric      | Baseline            | Full Fine-Tuning  | LoRA Fine-Tuning  |
|-------------|---------------------|-------------------|-------------------|
| ROUGE-1     | 0.4697              | 0.4854            | **0.4878**            |
| ROUGE-2     | 0.2288              | **0.2451**            | 0.2436            |
| ROUGE-L     | 0.3928              | **0.4057**            | 0.4043            |
| BLEU        | 0.1667              | 0.1773            | **0.1812**            |


---

### 📌 Notes on Evaluation Results

While **LoRA fine-tuning slightly outperformed full fine-tuning on ROUGE-1 and BLEU**, **full fine-tuning yielded better ROUGE-2 and ROUGE-L scores**. However, these results should be interpreted with caution:

> ⚠️ **Disclaimer:**
> The `samsum` dataset is relatively small, and this experiment is not extensively tuned. In general, **full fine-tuning tends to outperform parameter-efficient methods like LoRA**, especially on larger datasets and more complex tasks. The observed results may be influenced by factors such as limited training data, early stopping, or hyperparameter choices.

These experiments are intended to demonstrate **LoRA's potential as a lightweight fine-tuning strategy**, especially in resource-constrained settings.

---

## 📁 Notebooks Overview

| Notebook                       | Description                              |
| ------------------------------ | ---------------------------------------- |
| `base_line_evaluation.ipynb`  | Runs inference on the base FLAN-T5 model |
| `full_fine_tuning_LLMs.ipynb`      | Full model fine-tuning + evaluation      |
| `peft_lora_fine_tuning_LLMs.ipynb` | LoRA-based fine-tuning + evaluation      |

---

## 🚀 Reproducibility

1. Clone the repo
2. Open the notebook of your choice in Google Colab
3. Run all cells to reproduce results

---

## 📌 Notes

* All evaluations were performed on the test split of `knkarthick/samsum`.
* Training was done using PyTorch with Hugging Face Transformers and PEFT libraries.
* PEFT allowed faster training with minimal parameter updates (0.36% of total parameters).

---

## 🙌 Acknowledgements

* [Hugging Face Transformers](https://github.com/huggingface/transformers)
* [PEFT by Hugging Face](https://github.com/huggingface/peft)
* [DeepLearning.AI Generative AI with Large Language Models Course](https://www.deeplearning.ai)


