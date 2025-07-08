# KubeSecAgent

🔐 An LLM-powered agent that detects, tags, and explains security misconfigurations in Kubernetes manifests using Retrieval-Augmented Generation (RAG), with evaluation against industry-standard tools like Checkov, Terrascan, and KubeLinter.

---

## ✨ Features

- 🤖 **Agent-based Reasoning**: Uses a LangGraph-driven agent to interpret misconfiguration summaries.
- 📚 **RAG Integration**: Retrieves relevant misconfiguration tags using Chroma and OpenAI embeddings.
- 🧠 **LLM Tagging**: Leverages GPT-4o (or Gemini) to tag and explain Kubernetes issues.
- 🔍 **Benchmarking Support**: Evaluates performance against Checkov, Terrascan, and KubeLinter using precision/recall/F1.
- 📁 **Tool-specific Context**: Configurable to work with different tool-specific RAG databases.

---

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/KubeSecAgent.git
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
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key  # if using Gemini
TAVILY_API_KEY=your-tavily-key  # if using Tavily
```

## 📊 Evaluation

Use `evaluate_results.py` to compare the agent's output with labels from
industry tools. The script prints precision, recall, and F1 along with the
average number of extra tags (false positives) and missing tags (misses) per
configuration for each tool.

We compared their performance using the common classification metrics of
precision, recall, and the F1, weighted by the occurrence of the various KCF
misconfigs in the test set. For convenience, in the remainder of the paper the
terms precision, recall, and F1 represent the weighted precision, weighted
recall, and weighted F1, respectively. To compute the value of these
classification performance metrics, we define their building blocks as
follows:

- **True positives (TPs)** – Instances where both the LLM and at least one RB
  tool correctly detected a misconfig.
- **False positives (FPs)** – Instances where the LLM erroneously detected a
  misconfig that was not recognized by any of the RB tools. Because rule-based
  detectors are limited to their programmed rules, a FP might actually be a
  variant of a known misconfig that does not exactly match any coded rule.
- **False negatives (FNs)** – Instances where a misconfig was overlooked by the
  LLM but detected by one of the RB tools.
- **True negatives (TNs)** – Instances in which there was consensus between the
  LLM and the RB tools as to the absence of misconfig.

```bash
python evaluate_results.py path/to/agent_results.json
```