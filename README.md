# Proactive Web Agents with Interactive Multimodal Clarification

<p align="center">
<a href="https://ymwangv.github.io/">Yingming Wang</a> · 
<a href="https://yfyuan01.github.io/">Yifei Yuan</a> · 
<a href="https://anderssoegaard.github.io/">Anders Søgaard</a> · 
<a href="https://dengyang17.github.io/">Yang Deng</a>
</p>

---

## 🤗 Dataset
<h3 __align__="center"><a href="https://huggingface.co/datasets/ymwangv/MC-Mind2Web">https://huggingface.co/datasets/ymwangv/MC-Mind2Web</a></h3>

---

## 📋 Overview

Web agents powered by large language models (LLMs) and large multimodal models (LMMs) have demonstrated remarkable abilities in fulfilling user tasks through step-by-step planning and execution over the multimodal web environment. However, their effectiveness is limited in existing benchmarks: user instructions are assumed fully-specified and executable. In real-world scenarios, instructions are often underspecified, leaving agents unable to proceed without first seeking clarification from the user. To address this gap, we introduce **Multimodal Proactive Web Navigation**, a new task that requires an agent to identify underspecified details that prevent task completion, ask clarification questions to resolve these issues, and only then proceed with planning and execution. We present the **Multimodal Clarification Mind2Web** (**MC-Mind2Web**) dataset, constructed from the Mind2Web dataset. To benchmark the task, we propose **ProAct**, a dual-agent framework where a clarification agent detects the clarification need and asks clarification questions, while a navigation agent handles subsequent planning and execution based on the clarification results. Comprehensive experiments on MC-Mind2Web reveal the challenges of this problem. 

<p align="center">
  <img src="assets/annotation_pipeline.png" width="100%"/>
</p>

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/MC-Mind2Web.git
cd MC-Mind2Web
```

### Download Dataset

Download the MC-Mind2Web dataset from [HuggingFace](https://huggingface.co/datasets/ymwangv/MC-Mind2Web) and place it in the `data/` directory:
```bash
data/
├── train/
│   ├── dataset/
│   └── screenshots/
├── test_task/
├── test_website/
└── test_domain/
```

---

## 📚 Usage

### 1. Clarification

#### Prompting-based Approaches

**Run inference:**
```bash
python src/clarification/prompting/runner.py [args]
```

**Evaluate results:**
```bash
python src/clarification/prompting/evaluate.py [args]
```

#### Fine-tuning Approaches

**Train text model:**
```bash
python src/clarification/training/train_text.py [args]
```

**Train multimodal model:**
```bash
python src/clarification/training/train_llava.py [args]
```

**Evaluate models:**
```bash
# Text model
python src/clarification/training/evaluate_text.py [args]

# Multimodal model
python src/clarification/training/evaluate_llava.py [args]
```

---

### 2. Navigation

#### Candidate Generation

**Train candidate generation model:**
```bash
python src/candidate/train.py [args]
```

**Evaluate candidate generation:**
```bash
python src/candidate/evaluate.py [args]
```

#### Action Prediction

**Train action prediction model:**
```bash
python src/navigation/training/train.py [args]
```

**Evaluate action prediction:**
```bash
python src/navigation/training/evaluate.py [args]
```

**Note:** All evaluation results will be saved in the `eval/` directory.

---

## 📜 License

- **Code**: This project is licensed under the [MIT License](LICENSE). Parts of our code are adapted from [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) (MIT License).
- **Dataset**: The MC-Mind2Web dataset is released under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).