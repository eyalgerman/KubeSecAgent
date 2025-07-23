# KubeSecAgent

🔐 **KubeSecAgent** is an agent-based system for automatic detection and explanation of Kubernetes security misconfigurations. Built on **LangGraph**, it combines large language model (LLM) reasoning (via GPT-4o or similar), Retrieval-Augmented Generation (RAG), and automated validation to deliver superior coverage and clarity over traditional static analysis tools.  

**Paper:** [KubeSecAgent: Agent-Based Detection and Explanation of Kubernetes Security Misconfigurations](https://github.com/eyalgerman/KubeSecAgent)

---

## ✨ Key Features

- **Agent-Based Orchestration**: Multi-stage pipeline built with LangGraph, orchestrating summarization, retrieval, expert reasoning, and validation nodes.
- **LLM-Driven Tagging**: Uses GPT-4o (recommended) to interpret Kubernetes manifests, summarize potential issues, and tag misconfigurations.
- **Retrieval-Augmented Generation**: Queries ChromaDB vector stores (populated from Checkov, KubeLinter, Terrascan taxonomies) to ground LLM decisions in authoritative definitions.
- **Automated Validation Node**: Optional secondary LLM filter to suppress spurious predictions and enhance precision.
- **Benchmarking and Evaluation**: Directly benchmarks against industry-standard tools, reporting both corpus-level and per-manifest metrics.
- **Interpretable and Auditable**: Every stage is recorded in a state dictionary, with outputs serialized to JSONL for full auditability.

---

## 🏗️ Architecture

**Pipeline Overview:**

1. **Summarizer Module**: Generates concise summaries of potential misconfigurations from raw YAML.
2. **k-expert Node**: Receives YAML, the summary, and retrieved definitions; selects the most precise misconfiguration tags.
3. **Tool Node (RAG)**: Performs vector similarity search over ChromaDB to retrieve tool-specific context (Checkov, KubeLinter, Terrascan).
4. **Validator Node**: (Optional, recommended) Validates selected tags against the manifest and summary to reduce false positives.

**Recommended Model:**  
- **GPT-4o** (best balance of accuracy and context-aware reasoning).  
Other models (Gemini, Mistral-7B, GPT-4o-mini) were evaluated but showed lower robustness or could not reliably interact with the RAG module.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/eyalgerman/KubeSecAgent.git
cd KubeSecAgent
```

### 2. Install dependencies

```bash
pip install langgraph langchain langchain-openai
pip install langchainhub langchain-community langchain-text-splitters langchain-core langchain-cli openai transformers accelerate bitsandbytes tiktoken chromadb faiss-cpu
pip install langchain-google-genai google-generativeai
pip install chromadb
pip install pandas scikit-learn
```

Make sure you have `.env` with the following keys:

```env
OPENAI_API_KEY=your-openai-key  # if using OpenAI models
GOOGLE_API_KEY=your-google-key  # if using Gemini
HUGGINGFACE_API_KEY=your-huggingface-key  # if using HuggingFace models
TAVILY_API_KEY=your-tavily-key  # for use external knowledge base
```

---
## 🚀 Running KubeSecAgent

KubeSecAgent is run via `main.py` and is fully configurable via command-line arguments.

### **Basic Example (Recommended)**

To process **5 Kubernetes manifests** using the default OpenAI GPT-4o model and `checkov` tags:

```bash
python main.py --limit 5
```

### **Full Command-Line Usage**

```bash
python main.py \
  --filename PATH_TO_MANIFESTS.json \
  --tags-tools checkov kube_linter terrascan \
  --limit 100 \
  --top-k 4 \
  --model-provider openai \
  --model-name gpt-4o \
  --no-validation   # Add this flag to disable the validator node (optional)
```

#### **Arguments Explained:**

| Argument               | Description                                                                                | Default            | Example Value                                         |
|------------------------|--------------------------------------------------------------------------------------------|--------------------|-------------------------------------------------------|
| `--filename`           | Path to JSON file containing Kubernetes manifests                                          | `10k_open_source_manifests.json` | `my_manifests.json`                                   |
| `--tags-tools`         | One or more tools to use for RAG database (space separated: `checkov kube_linter terrascan`) | `['checkov']`      | `checkov kube_linter`                                 |
| `--limit`              | Maximum number of manifests to process per tool (`-1` for all)                             | `5`                | `100` or `-1`                                         |
| `--top-k`              | Number of top results to retrieve from the RAG tool                                        | `5`                | `4`                                                   |
| `--model-provider`     | Language model provider (`openai`, `google`, `huggingface`)                                | `openai`           | `google` or `huggingface`                             |
| `--model-name`         | Model name for the provider (see notes below)                                              | `gpt-4o`           | `gpt-4o-mini`, `gemini-1.5-pro`, HuggingFace model ID |
| `--no-validation`      | (Optional) If set, disables the validator node (by default, validation is enabled)         | *(enabled)*        | *(just include flag to disable)*                      |

### **Provider & Model Selection**

- **OpenAI**: `--model-provider openai --model-name gpt-4o`
- **Google**: `--model-provider google --model-name gemini-1.5-pro`
- **HuggingFace**: `--model-provider huggingface --model-name mistralai/Mistral-7B-Instruct-v0.2`

## **Requirements for input and output**

- Models: See [Setup](#setup) for API keys and dependencies.
- Input: The input file should be a JSON array, each element with fields:
  - `file`: filename
  - `file_content`: YAML as string
- Results are saved in a timestamped JSON file in the `results/` directory.
- After processing, the script automatically evaluates the agent’s output against the reference tool (precision, recall, F1, extras, misses).


---

## 📈 Evaluation & Metrics

KubeSecAgent was evaluated on a benchmark of **2,000 real-world Kubernetes manifests** against Checkov, KubeLinter, and Terrascan. The results were measured by the following metrics:

- **Precision / Recall / F1**: (Weighted and Macro) Based on true/false positives and negatives compared to reference tool annotations.
- **Extras**: Average extra tags per manifest (potential false positives).
- **Misses**: Average ground-truth tags missed per manifest (potential false negatives).

**Sample Results** (using GPT-4o, validator node enabled, k=4):

| Tool       | Weighted Precision | Weighted Recall | Weighted F1 | Extras ↓ | Misses ↓ |
| ---------- | ------------------ | --------------- | ----------- | -------- | -------- |
| Checkov    | 0.706              | 0.313           | 0.434       | 1.53     | 8.08     |
| KubeLinter | 0.776              | 0.275           | 0.407       | 0.81     | 7.37     |
| Terrascan  | 0.785              | 0.452           | 0.573       | 1.10     | 4.84     |

Ablation studies on retrieval depth (`k`) and validator node inclusion demonstrate that expanding RAG context improves recall and F1, while the validator node reduces false positives.

---

## 👨‍💻 Our Team

- **Eyal German** - [germane@post.bgu.ac.il](mailto\:germane@post.bgu.ac.il)
- **Ron Solomon** - [ronso@post.bgu.ac.il](mailto\:ronso@post.bgu.ac.il)
- **Nir Aharoni** - [nirahar@post.bgu.ac.il](mailto\:nirahar@post.bgu.ac.il)

Department of Software and Information Systems Engineering,\
Ben-Gurion University of the Negev, Israel