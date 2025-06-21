import os
import json
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from tqdm import tqdm
from evaluate_results import evaluate_llm_per_tool, evaluate_llm_per_tool_with_normalize
from tools import summarize_yaml, search_tool, build_rag_tool
from rag import start_rag
import nodes

load_dotenv()

def setup_llm():
    """Return the OpenAI chat model used by the agent."""

    # Uncomment below to use Gemini instead of OpenAI
    # from langchain_google_genai import ChatGoogleGenerativeAI
    # return ChatGoogleGenerativeAI(
    #     model="gemini-1.5-flash-latest",
    #     api_key=os.getenv("GOOGLE_API_KEY")
    # )
    return ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))


def build_langgraph_app(llm_with_tools, tools):
    """Compile a simple LangGraph graph for running the agent.

    Args:
        llm_with_tools: Chat model with tools bound.
        tools (list): Tool callables to expose.

    Returns:
        The compiled graph instance.
    """

    graph = StateGraph(nodes.AgentState)
    graph.set_entry_point("k_expert")
    graph.add_node("k_expert", nodes.k_expert(llm_with_tools))
    # define graph nodes
    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("k_expert", nodes.tool_use)
    graph.add_edge("tools", "k_expert")
    return graph.compile()

def build_langgraph_app_with_validator(llm_with_tools, tools, tag_definitions, llm=None):
    """Compile a LangGraph graph with a validator node.

    Args:
        llm_with_tools: Chat model with tools bound.
        tools (list): Tool callables to expose.
        tag_definitions (dict): Mapping of tag -> description for validation.
        llm: Optional chat model for the validator.

    Returns:
        The compiled graph instance.
    """

    graph = StateGraph(nodes.AgentState)
    graph.set_entry_point("k_expert")

    graph.add_node("k_expert", nodes.k_expert(llm_with_tools))
    graph.add_node("validator", nodes.TagValidator(llm, tag_definitions))

    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)

    # -- existing edges
    graph.add_conditional_edges("k_expert", nodes.tool_use)
    graph.add_edge("tools", "k_expert")

    # -- new validation edge
    graph.add_edge("k_expert", "validator")
    graph.set_finish_point("validator")   # graph ends here
    return graph.compile()



def process_configs(app, configs, llm):
    """Run the LangGraph app over a list of configs and collect tag results.

    Args:
        app: Compiled LangGraph app.
        configs (list[dict]): List of file entries with ``file`` and ``file_content``.
        llm: Chat model used for YAML summarization.

    Returns:
        List of dicts with ``file_name`` and extracted ``tags``.
    """

    all_results = []

    for config in tqdm(configs, desc="Processing configs"):
        yml = config["file_content"]
        file_tags = []

        for missconfig in summarize_yaml(yml, llm):
            state = {
                "messages": ["start"],
                "yaml": yml,
                "summary": missconfig,
                "tags": []
            }
            result = app.invoke(state)
            file_tags.extend(result["tags"])

        all_results.append({
            "file_name": config["file"],
            "tags": file_tags
        })
        print(f"Processed {config['file']} with tags: {file_tags}\n\n")

    return all_results


def save_results(results, out_dir="results", tags_tool="checkov"):
    """Persist results to a timestamped JSON file and return its path.

    Args:
        results (list): Output from :func:`process_configs`.
        out_dir (str): Directory to write results into.
        tags_tool (str): Name of the tool used for tagging.

    Returns:
        Path to the saved JSON file.
    """

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(out_dir, f"agent_results_tags_{tags_tool}_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return output_path


def run_agent_for_tool(filename="examples.json", tags_tool="checkov", limit=-1, with_validation=True):
    """Run the agent on a single tool's dataset and evaluate the results.

    Args:
        filename (str): JSON file containing configs to process.
        tags_tool (str): Tool name whose tag database is used.
        limit (int): Maximum number of configs to process; ``-1`` means all.
        with_validation (bool): Whether to run the validator node.

    Returns:
        None. Results are saved to disk and printed to stdout.
    """

    if not os.path.exists(f"rag_db_{tags_tool}"):
        start_rag(tags_tool)
    else:
        print("RAG already initialized. Skipping setup.")

    llm = setup_llm()
    rag = build_rag_tool(tags_tool)
    tools = [search_tool, rag]
    llm_with_tools = llm.bind_tools(tools=tools)
    if with_validation:
        tag_defs = json.load(open("misconfigs_map.json")).get(tags_tool, {})
        app = build_langgraph_app_with_validator(llm_with_tools, tools, tag_definitions=tag_defs, llm=llm)
    else:
        app = build_langgraph_app(llm_with_tools, tools)

    with open(filename, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if limit > 0:
        configs = configs[:limit]

    results = process_configs(app, configs, llm)
    result_path = save_results(results, tags_tool=tags_tool)
    # evaluate results
    print("Evaluating without normalization:")
    evaluate_llm_per_tool(result_path, tools_to_compare=[tags_tool, tags_tool + "_new"])
    print("Evaluating with normalization:")
    evaluate_llm_per_tool_with_normalize(result_path, tools_to_compare=[tags_tool, tags_tool + "_new"])


def main(filename="examples.json", tags_tools=None, limit=-1, with_validation=True):
    """Entry point for running the agent over multiple tagging tools.

    Args:
        filename (str): JSON file of configs.
        tags_tools (list[str] or None): Tools to run; defaults to ["checkov"].
        limit (int): Maximum configs to process per tool.
        with_validation (bool): Whether to enable result validation.
    """

    if tags_tools is None:
        tags_tools = ["checkov" ]#, "kube_linter", "terrascan"]
        # tags_tools = ["kube_linter", "terrascan"]
    for tool in tqdm(tags_tools, desc="Running agents for tools"):
        try:
            print(f"Running agent for tool: {tool}")
            run_agent_for_tool(filename, tool, limit, with_validation=with_validation)
            print(f"Finished processing with tool: {tool}")
        except Exception as e:
            print(f"Error processing tool {tool}: {e}")
            continue


if __name__ == "__main__":
    main(filename="10k_open_source_manifests.json", limit=5, with_validation=True)
