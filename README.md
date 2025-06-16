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