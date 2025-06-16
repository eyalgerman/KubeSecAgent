import os
import json
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from tqdm import tqdm

from evaluate_results import evaluate_llm_per_tool
from tools import summarize_yaml, search_tool, build_rag_tool
from rag import start_rag
import nodes

load_dotenv()

def setup_llm():
    # Uncomment below to use Gemini instead of OpenAI
    # from langchain_google_genai import ChatGoogleGenerativeAI
    # return ChatGoogleGenerativeAI(
    #     model="gemini-1.5-flash-latest",
    #     api_key=os.getenv("GOOGLE_API_KEY")
    # )
    return ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))


def build_langgraph_app(llm_with_tools, tools):
    graph = StateGraph(nodes.AgentState)
    graph.set_entry_point("k_expert")
    graph.add_node("k_expert", nodes.k_expert(llm_with_tools))
    # define graph nodes
    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("k_expert", nodes.tool_use)
    graph.add_edge("tools", "k_expert")
    return graph.compile()


def process_configs(app, configs, llm):
    all_results = []

    for config in tqdm(configs, desc="Processing configs"):
        yml = config["file_content"]
        file_tags = []

        for missconfig in summarize_yaml(yml, llm):
            state = {
                "messages": ["start"],
                "yaml": yml,
                "summary": missconfig,
                "tag": None
            }
            result = app.invoke(state)
            file_tags.append(result["tag"])

        all_results.append({
            "file_name": config["file"],
            "tags": file_tags
        })

    return all_results


def save_results(results, out_dir="results", tags_tool="checkov"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(out_dir, f"agent_results_tags_{tags_tool}_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return output_path


def run_agent_for_tool(filename="examples.json", tags_tool="checkov", limit=-1):

    if not os.path.exists(f"rag_db_{tags_tool}"):
        start_rag(tags_tool)
    else:
        print("RAG already initialized. Skipping setup.")

    llm = setup_llm()
    rag = build_rag_tool(tags_tool)
    tools = [search_tool, rag]
    llm_with_tools = llm.bind_tools(tools=tools)
    app = build_langgraph_app(llm_with_tools, tools)

    with open(filename, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if limit > 0:
        configs = configs[:limit]

    results = process_configs(app, configs, llm)
    result_path = save_results(results, tags_tool=tags_tool)
    # evaluate results
    evaluate_llm_per_tool(result_path, tools_to_compare=[tags_tool, tags_tool + "_new"])


def main(filename="examples.json", tags_tools=None, limit=-1):
    if tags_tools is None:
        tags_tools = ["checkov", "kube_linter", "terrascan"]
        tags_tools = ["kube_linter", "terrascan"]
    for tool in tqdm(tags_tools, desc="Running agents for tools"):
        print(f"Running agent for tool: {tool}")
        run_agent_for_tool(filename, tool, limit)
        print(f"Finished processing with tool: {tool}")


if __name__ == "__main__":
    main(limit=3)
