# PyTorch LoRA Implementation for MNIST Fine-Tuning

This repository contains a Jupyter Notebook that demonstrates a practical, from-scratch implementation of **Low-Rank Adaptation (LoRA)** using PyTorch. The project showcases how to efficiently fine-tune a pre-trained neural network for a specific task without modifying the original model's weights.

The core idea is demonstrated by:
1.  Pre-training a large neural network on the full MNIST dataset.
2.  Identifying a class the model performs poorly on (the digit '9').
3.  Using LoRA to fine-tune the model *only* on images of the digit '9'.
4.  Showing the significant performance improvement on the target class while preserving the original model's integrity.

---

## ✨ Key Concepts Demonstrated

* **LoRA from Scratch**: The notebook implements the LoRA logic in a custom `nn.Module`, showing how the weight update `ΔW` is decomposed into two smaller, low-rank matrices (`A` and `B`).
* **Parameter-Efficient Fine-Tuning (PEFT)**: It highlights the core benefit of LoRA—fine-tuning a model by training only a tiny fraction of new parameters. In this example, the number of trainable parameters is increased by only **0.242%**.
* **PyTorch Parametrizations**: The implementation uses `torch.nn.utils.parametrize` to non-destructively apply the LoRA matrices to the existing `nn.Linear` layers. This is a clean and modern way to modify a layer's behavior without changing its underlying code.
* **Non-Invasive Training**: The original weights of the pre-trained model are frozen and remain completely unchanged during fine-tuning. This is explicitly verified using assertions.
* **Switchable Adapters**: The notebook includes a helper function to easily **enable and disable** the LoRA adaptation, allowing the model to instantly switch between its general and specialized behaviors.

---

## ⚙️ How It Works

The notebook follows a clear, step-by-step process:

1.  **Base Model Training**: A large neural network (`RichBoyNet`) is trained for a single epoch on the entire MNIST training set to simulate a general-purpose, pre-trained model.
2.  **Baseline Evaluation**: The model's initial performance is tested. It achieves an overall accuracy of **95.3%** but makes **107 mistakes** on the digit '9', identifying it as a weakness. The original weights are saved for later comparison.
3.  **LoRA Implementation**:
    * A `LoRAParametrization` class is defined. It contains two low-rank matrices: `lora_A` and `lora_B`.
    * Its `forward` method implements the core LoRA equation: $W_{new} = W_{original} + (B \cdot A) \cdot \frac{\alpha}{r}$.
    * This module is registered to the `weight` of each `nn.Linear` layer in the network using `parametrize.register_parametrization`.
4.  **Fine-Tuning on a Subset**:
    * The original parameters of the network (weights and biases) are **frozen** so they cannot be updated.
    * The training dataset is filtered to contain *only* images of the digit '9'.
    * The model is trained for a small number of steps (100 batches). Only the LoRA matrices `A` and `B` are trained.
5.  **Final Evaluation**: The model is evaluated in two states to demonstrate LoRA's impact.

---

## 📊 Results

The results clearly show the effectiveness of LoRA for specialized tasks.

| Model State                        | Overall Accuracy | Errors on Digit '9' |
| :--------------------------------- | :--------------: | :-----------------: |
| **Before Fine-Tuning** (Original)  | 95.3%            | 107                 |
| **After Fine-Tuning (LoRA Enabled)** | 84.3%            | **11** |
| **After Fine-Tuning (LoRA Disabled)** | 95.3%            | 107                 |

As shown above:
* With LoRA **enabled**, the model becomes an expert on the digit '9', reducing its errors from 107 to just **11**. The overall accuracy drops, which is expected since the model is now heavily biased toward a single class.
* With LoRA **disabled**, the model's performance is identical to its original state, proving that the fine-tuning process did not alter the base weights.

---
