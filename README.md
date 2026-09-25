# CoLoRA: Parameter-Efficient Fine-Tuning for Convolutional Models

CoLoRA (**Convolutional Low-Rank Adaptation**) is a parameter-efficient fine-tuning (PEFT) method for convolutional neural networks. It adapts pretrained convolutional kernels through structured pointwise-depthwise updates and can merge the learned update into the original convolutional weights before deployment.

This repository contains a clean VGG16-CoLoRA implementation for **OCTMNISTv2**, together with a tutorial notebook and a command-line training/evaluation script.

> **Paper / preprint**  
> Mariano Rivera and Angello Hoyos. *COLORA: Efficient Fine-Tuning for Convolutional Models with a Study Case on Optical Coherence Tomography Image Classification*.  
> arXiv:2505.18315 — https://arxiv.org/abs/2505.18315

## Highlights

- Convolution-specific PEFT using **1×1 pointwise** and **depthwise spatial** updates.
- More than **80% reduction in trainable convolutional-update parameters** relative to full convolutional fine-tuning.
- Learned CoLoRA updates can be **merged into the pretrained convolutional kernels**, preserving the base inference graph.
- Support for **full**, **balanced**, and **distilled** OCTMNISTv2 training data.
- Validation-based model selection with the **held-out test split reserved for final evaluation**.
- Training/evaluation script plus a tutorial-oriented notebook.
- Experiments in the revised manuscript include VGG16, ResNet50, additional MedMNIST datasets, CIFAR-100, and ImageNet-R.

## Repository structure

```text
CoLoRA/
├── README.md
├── requirements.txt
├── VGG16_CoLoRA_OCTMNIST_tutorial.ipynb
├── vgg16_colora_octmnist.py
└── imgs/
```

The distilled dataset file is not required for the `full` or `balanced` modes. To use `distilled`, place your precomputed file in the repository (or provide its path):

```text
OCTMNISTv2_Distilled.npz
```

The file must contain arrays named `X` and `Y`.

## CoLoRA formulation

For a pretrained convolutional kernel \(K_0\), CoLoRA learns a structured update

\[
K = K_0 + \Delta K,
\]

where the update is represented using pointwise and depthwise components. The pretrained convolution is frozen during CoLoRA adaptation, while the lightweight update branch is optimized.

After adaptation, the learned update can be fused into the original convolutional kernel. This keeps the deployed model free of an additional residual CoLoRA branch.

For VGG16, the implementation explicitly separates convolution and ReLU layers, initializes the backbone from ImageNet weights, and uses the following classification head:

```text
VGG16 backbone
    ↓
1×1 Conv2D (512 → 128)
    ↓
Flatten
    ↓
Dense(128, ReLU)
    ↓
Dense(4, logits)
```

The four OCTMNIST classes are:

1. Choroidal Neovascularization (ChN)
2. Diabetic Macular Edema (DME)
3. Drusen
4. Normal

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/ajhoyos/CoLoRA.git
cd CoLoRA

python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU support depends on the TensorFlow build and your CUDA/ROCm environment. The repository itself does not install vendor-specific GPU drivers or toolkits.

## Dataset modes

The new implementation exposes three OCTMNISTv2 training modes.

### Full

Uses the complete original OCTMNISTv2 training split.

```bash
python vgg16_colora_octmnist.py --dataset-mode full
```

### Balanced

Builds a deterministic class-balanced training subset using the size of the smallest class. For OCTMNISTv2 this corresponds to **7,754 images per class**, or **31,016 training images**.

```bash
python vgg16_colora_octmnist.py --dataset-mode balanced
```

### Distilled

Loads the precomputed distilled dataset:

```bash
python vgg16_colora_octmnist.py \
    --dataset-mode distilled \
    --distilled-path ./OCTMNISTv2_Distilled.npz
```

In the revised manuscript, the distilled subset was constructed class-by-class from the balanced set by sorting samples in ascending predictive entropy, removing ranks **1–689** and **7,730–7,754**, and retaining ranks **690–7,729**. This results in **7,040 samples per class**.

The current notebook and script **load** the distilled subset; they do not repeat the entropy-ranking/distillation procedure.

## Training and evaluation

The command-line script is the recommended entry point for reproducible training and evaluation.

### Train and evaluate

```bash
python vgg16_colora_octmnist.py
```

Default configuration:

```text
Dataset mode:       distilled
Image size:         224 × 224
Batch size:         8
Epochs:             20
Backbone weights:   ImageNet
CoLoRA placement:   VGG16 blocks 2–5
Learning rate:      1e-4
Weight decay:       1e-5
Augmentation:       enabled
```

### Balanced training

```bash
python vgg16_colora_octmnist.py \
    --dataset-mode balanced \
    --epochs 20
```

### Full training set

```bash
python vgg16_colora_octmnist.py \
    --dataset-mode full \
    --epochs 20
```

### Disable augmentation

```bash
python vgg16_colora_octmnist.py --no-augmentation
```

### Change CoLoRA placement

For example, to adapt only blocks 3–5:

```bash
python vgg16_colora_octmnist.py \
    --colora-blocks block3_conv block4_conv block5_conv
```

### Train only

```bash
python vgg16_colora_octmnist.py --run-mode train
```

### Evaluate an existing model

```bash
python vgg16_colora_octmnist.py \
    --run-mode eval \
    --model-path ./results/distilled_blocks_2_3_4_5/best_validation_model.keras
```

Use

```bash
python vgg16_colora_octmnist.py --help
```

to see the complete command-line interface.

## Output files

A training run creates an experiment directory such as

```text
results/distilled_blocks_2_3_4_5/
```

containing:

```text
best_validation_model.keras
training_history.csv
training_curves.png
training_metadata.json
experiment_config.json
dataset_metadata.json
test_summary.csv
test_class_metrics.csv
test_confusion_matrix.csv
test_confusion_matrix.png
```

The test split is loaded only during final evaluation.

## Tutorial notebook

`VGG16_CoLoRA_OCTMNIST_tutorial.ipynb` presents the same main workflow in notebook form:

1. experiment configuration;
2. OCTMNIST loading;
3. TensorFlow data pipeline;
4. VGG16 construction;
5. CoLoRA layers and placement;
6. training using validation-based checkpoint selection;
7. final test inference and metrics.

For manuscript-scale experiments, use the full epoch schedule. If the notebook training call has been temporarily set to a small value such as `epochs=3` for a smoke test, change it back to:

```python
epochs=EPOCHS
```

where `EPOCHS = 20`.

## Reference results from the revised manuscript

The values below summarize the revised manuscript and are provided as **reference experimental results**. They should not be interpreted as a guarantee that every single run will reproduce exactly the same values.

### VGG16-CoLoRA layer placement on OCTMNISTv2

| Method / placement | Trainable params | Avg. epoch time | Test accuracy (mean ± SD) | Max. test accuracy |
|---|---:|---:|---:|---:|
| CoLoRA blocks 1–5 | 2.55M | 4.42 min | 0.956 ± 0.003 | 0.962 |
| **CoLoRA blocks 2–5** | **2.54M** | **3.00 min** | **0.959 ± 0.003** | **0.966** |
| CoLoRA blocks 3–5 | 2.51M | 2.79 min | 0.957 ± 0.002 | 0.961 |
| CoLoRA blocks 4–5 | 2.34M | 2.15 min | 0.940 ± 0.003 | 0.957 |
| CoLoRA block 5 | 1.67M | 1.64 min | 0.941 ± 0.002 | 0.949 |
| CoLoRA blocks 2–5, pointwise only | 0.91M | 1.95 min | 0.933 ± 0.005 | 0.944 |
| Transfer learning | 0.87M | 1.32 min | 0.894 ± 0.007 | 0.908 |
| Full fine-tuning | 15.58M | 3.27 min | 0.918 ± 0.005 | 0.957 |

The placement study indicates that blocks **2–5** provide the strongest predictive result among the evaluated VGG16 placements, while blocks **3–5** provide a closely related result with somewhat lower training cost.

### Distilled OCTMNISTv2

For the distilled VGG16-CoLoRA experiment, the revised manuscript reports a best test accuracy of **0.966** and an AUC of approximately **0.995**. The class-wise distilled experiment reports the following averages:

| Metric | Average |
|---|---:|
| AUC | 0.995 |
| Recall | 0.962 |
| Precision | 0.963 |
| F1 | 0.962 |

### Controlled PEFT comparison on balanced OCTMNISTv2

| Method | Test AUC | Test accuracy | Trainable params |
|---|---:|---:|---:|
| Transfer Learning | 0.979 | 0.933 | 0.87M |
| Adapters | 0.995 | 0.957 | 1.50M |
| BitFit | 0.979 | 0.896 | 0.87M |
| Conv-LoRA R4, α=8 | 0.991 | 0.944 | 0.90M |
| Conv-LoRA R8, α=16 | 0.990 | 0.943 | 0.93M |
| Conv-LoRA R16, α=32 | 0.993 | 0.949 | 0.99M |
| CoLoRA block 5 | 0.993 | 0.939 | 1.67M |
| **CoLoRA blocks 2–5** | **0.994** | **0.954** | **2.54M** |

Adapters obtained the strongest raw balanced-dataset accuracy in this controlled comparison. CoLoRA blocks 2–5 outperformed the evaluated Conv-LoRA configurations while retaining a mergeable convolution-aware update and avoiding an explicit rank/scaling-factor hyperparameter.

### Additional MedMNIST datasets

VGG16-CoLoRA improved over VGG16 transfer learning in the reported experiments:

| Dataset | Transfer learning AUC / Acc. | CoLoRA AUC / Acc. |
|---|---:|---:|
| RetinaMNIST | 0.756 / 0.527 | 0.776 / 0.552 |
| PathMNIST | 0.981 / 0.876 | 0.993 / 0.956 |
| BloodMNIST | 0.995 / 0.932 | 0.998 / 0.981 |

### ResNet50

The revised manuscript reports **AUC 0.992** and **accuracy 0.951** for ResNet50-CoLoRA, using approximately **14.7M backpropagated parameters** compared with 37.7M parameters for the full ResNet50 model.

### Non-medical datasets

The revised manuscript also evaluates CoLoRA outside medical imaging.

#### CIFAR-100

| Method | AUC | Top-1 | Top-5 |
|---|---:|---:|---:|
| Transfer Learning | 0.889 ± 0.004 | 0.258 ± 0.002 | 0.535 ± 0.004 |
| Adapters | 0.942 ± 0.004 | 0.449 ± 0.004 | 0.730 ± 0.008 |
| BitFit | 0.914 ± 0.003 | 0.327 ± 0.003 | 0.613 ± 0.003 |
| Conv-LoRA | 0.947 ± 0.002 | 0.424 ± 0.006 | 0.732 ± 0.008 |
| **CoLoRA blocks 2–5** | **0.961 ± 0.001** | **0.451 ± 0.004** | **0.766 ± 0.003** |
| CoLoRA block 5 | 0.911 ± 0.004 | 0.324 ± 0.003 | 0.621 ± 0.009 |

#### ImageNet-R

For the controlled class-balanced ImageNet-R subset:

| Method | AUC | Top-1 | Top-5 |
|---|---:|---:|---:|
| Transfer Learning | 0.737 ± 0.009 | 0.087 ± 0.008 | 0.210 ± 0.009 |
| Conv-LoRA | 0.697 ± 0.016 | 0.071 ± 0.009 | 0.174 ± 0.014 |
| **CoLoRA blocks 2–5** | **0.754 ± 0.008** | **0.109 ± 0.005** | **0.241 ± 0.012** |

These experiments are cross-domain validation experiments rather than exhaustive optimization of CIFAR-100 or ImageNet-R.

## Training-memory note

Parameter efficiency does not imply an equivalent reduction in peak training memory.

Under the GPU-memory protocol reported in the revised manuscript:

- full VGG16 fine-tuning: **3.320 GiB**;
- transfer learning: **2.316 GiB**;
- Conv-LoRA (r=8, α=16): **3.052 GiB**;
- CoLoRA blocks 2–5: **3.172 GiB**.

CoLoRA should therefore be characterized primarily as **parameter-efficient and deployment-efficient**. Peak training memory also depends on activation storage, optimizer states, spatial resolution, and which convolutional blocks are adapted.

## Implementation note versus manuscript experiments

The revised manuscript studies an optional **merge-reset** procedure in which CoLoRA updates may be merged into the backbone during training and the auxiliary update parameters reinitialized.

The cleaned repository implementation provided here prioritizes a simple validation-selected training/evaluation workflow: the CoLoRA branch is optimized, the checkpoint with the best validation accuracy is restored, and its learned update is merged before the saved model is evaluated on the held-out test set.

Consequently, the manuscript tables above are reference results from the experimental study rather than guaranteed outputs of a single default execution of the cleaned script.

## Reproducibility

For a new experiment, record at least:

- dataset mode;
- CoLoRA block placement;
- random seed;
- batch size;
- number of epochs;
- augmentation setting;
- TensorFlow/Keras versions;
- GPU model and software stack.

The training script stores the experiment configuration and dataset metadata alongside the saved model.

## Citation

If you use CoLoRA or this repository in academic work, please cite:

```bibtex
@article{rivera2025colora,
  title   = {COLORA: Efficient Fine-Tuning for Convolutional Models with a Study Case on Optical Coherence Tomography Image Classification},
  author  = {Rivera, Mariano and Hoyos, Angello},
  journal = {arXiv preprint arXiv:2505.18315},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.18315}
}
```

## Acknowledgment

The OCTMNIST experiments use the MedMNIST benchmark. Please also cite MedMNIST and the original OCT dataset as appropriate when using those data in published work.
