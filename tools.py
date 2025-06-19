import os
from langchain_community.tools import TavilySearchResults
from langchain.agents import tool
from dotenv import load_dotenv
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import List

load_dotenv()

# llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
#
# # Load Chroma DB once
# chroma_db = Chroma(
#     collection_name="k8s-misconfigs",
#     embedding_function=OpenAIEmbeddings()
# )


def build_rag_tool(tool_name: str, k: int = 5) -> tool:
    """
    Creates a RAG @tool instance for a specific tool (checkov, terrascan, kube_linter).
    """
    valid_tools = {"checkov", "terrascan", "kube_linter"}
    if tool_name not in valid_tools:
        raise ValueError(f"Invalid tool_name '{tool_name}'. Must be one of: {valid_tools}")

    @tool
    def rag(query: str) -> str:
        """
        Tool-specific RAG database query to return misconfiguration tags relevant for a Kubernetes config summary.
        """
        db = Chroma(
            collection_name=f"{tool_name}_misconfigs",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"rag_db_{tool_name}"
        )

        print(f"\n🔍 Querying RAG for tool: {tool_name}")
        print(f"Query: {query}\n")

        results = db.similarity_search(query, k=k)

        print(f"RAG results: {results}\n")

        if not results:
            return "No relevant misconfiguration tags found."

        formatted_results = []
        seen = set()
        for doc in results:
            tag = doc.metadata.get("tag", "N/A")
            content = doc.page_content.strip()
            if tag not in seen:
                seen.add(tag)
                formatted_results.append(f"{tag}: {content}")

        return "\n".join(formatted_results)

    return rag

# @tool
# def rag(query: str) -> str:
#     """
#     RAG database query to return misconfiguration tags relevant for a Kubernetes config summary.
#     """
#     db = Chroma(
#         collection_name="checkov_misconfigs",
#         embedding_function=OpenAIEmbeddings(),
#         persist_directory="rag_db"
#     )
#
#     print(f"\nQuery: {query}\n")
#
#     results = db.similarity_search(query, k=3)
#
#     print(f"\Rag result: {results}\n")
#     if not results:
#         print("No results found.")
#         return "No relevant misconfiguration tags found."
#
#     formatted_results = []
#     seen = set()
#     for i, doc in enumerate(results):
#         tag = doc.metadata.get("tag", "N/A")
#         content = doc.page_content.strip()
#         if tag not in seen:
#             seen.add(tag)
#             formatted_results.append(f"{tag}: {content}")
#     p = "\n".join(formatted_results)
#     print(p)
#
#     return p


tavily_key = os.getenv("TAVILY_API_KEY")
search_tool = TavilySearchResults(api_key=tavily_key)


def summarize_yaml(yaml: str, llm) -> List[str]:
    """Summarize YAML and return a list of security issues (or empty if none)."""
    
    prompt = f"""You are a Kubernetes security expert.

Analyze the following Kubernetes manifest and identify any potential security misconfigurations,
missing best practices, or confirm if it appears secure and well-structured.

find as much problems as possible 

Respond with a short bullet list of issues (e.g., '- Containers run as root'), or just say "No issues found."
Do not include explanations or suggestions.

Manifest:
{yaml}
"""

    # 1. Call LLM
    response = llm.invoke([HumanMessage(content=prompt)])

    # 2. Extract text
    summary_text = response.content.strip()
    print(f"🧠 Final summary used:\n{summary_text}\n")

    # 3. Convert to Python list
    if "no issues found" in summary_text.lower():
        return []

    # Split lines that begin with "-", "*" or are numbered
    issues = [
        line.strip("-*•1234567890. ").strip()
        for line in summary_text.splitlines()
        if line.strip() and not line.lower().startswith("no issues")
    ]
    print(issues)

    return issues
