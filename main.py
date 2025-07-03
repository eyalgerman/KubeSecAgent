import os
import json
import datetime
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# Local project imports
from evaluate_results import evaluate_llm_per_tool, evaluate_llm_per_tool_with_normalize
from tools import summarize_yaml, search_tool, build_rag_tool
from rag import start_rag
import nodes

load_dotenv()

# --- Constants for Graph Nodes ---
# Using constants for node names avoids typos and makes refactoring easier.
K_EXPERT_NODE = "k_expert"
TOOL_NODE = "tools"
VALIDATOR_NODE = "validator"


def setup_llm(model_provider: str, model_name: str):
    """
    Initializes and returns the specified chat model based on the provider.

    Args:
        model_provider (str): The provider ('openai', 'google', 'huggingface').
        model_name (str): The name of the model to initialize.

    Returns:
        An instance of a LangChain chat model.
    """
    print(f"Setting up LLM from provider '{model_provider}' with model '{model_name}'...")

    if model_provider == "openai":
        return ChatOpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))

    if model_provider == "google":
        return ChatGoogleGenerativeAI(model=model_name, api_key=os.getenv("GOOGLE_API_KEY"))

    if model_provider == "huggingface":
        try:
            import torch
            from transformers import pipeline
            from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
            from langchain_community.chat_models.huggingface import ChatHuggingFace

            print("Initializing Hugging Face model...")
            print("NOTE: The selected model MUST be fine-tuned for tool/function-calling.")

            # Create a text-generation pipeline with the specified model
            hf_pipeline = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance on modern GPUs
                device_map="auto",  # Automatically distribute the model across available hardware
            )

            # Wrap the pipeline in LangChain's LLM and ChatModel interfaces
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
            return ChatHuggingFace(llm=llm)

        except ImportError:
            print("Hugging Face libraries not found.")
            print("Please install them with: pip install torch transformers accelerate bitsandbytes langchain_community")
            raise
        except Exception as e:
            print(f"Error loading Hugging Face model '{model_name}': {e}")
            print("Please ensure the model name is correct and you have enough resources.")
            raise

    raise ValueError(f"Unsupported model provider: {model_provider}")


def build_graph(llm_with_tools, tools, validator_config: dict | None = None):
    """
    Compiles a LangGraph graph, with an optional validator node.
    This single function replaces the two separate build functions.

    Args:
        llm_with_tools: Chat model with tools bound.
        tools (list): Tool callables to expose.
        validator_config (dict | None): If provided, adds a validator node.
                                        Expected keys: "llm" and "tag_definitions".

    Returns:
        The compiled graph instance.
    """
    graph = StateGraph(nodes.AgentState)
    graph.set_entry_point(K_EXPERT_NODE)

    # Add core nodes
    graph.add_node(K_EXPERT_NODE, nodes.k_expert(llm_with_tools))
    graph.add_node(TOOL_NODE, ToolNode(tools=tools))

    # Add core edges
    graph.add_conditional_edges(K_EXPERT_NODE, nodes.tool_use)
    graph.add_edge(TOOL_NODE, K_EXPERT_NODE)

    # Conditionally add the validator node and its connections
    if validator_config:
        graph.add_node(
            VALIDATOR_NODE,
            nodes.TagValidator(validator_config["llm"], validator_config["tag_definitions"])
        )
        # The K_Expert's final answer is sent to the validator
        graph.add_edge(K_EXPERT_NODE, VALIDATOR_NODE)
        graph.set_finish_point(VALIDATOR_NODE)
    else:
        # If no validator, the graph finishes after the K_Expert gives a final answer
        graph.set_finish_point(K_EXPERT_NODE)

    return graph.compile()


def process_configs(app, configs: list[dict], llm):
    """Run the LangGraph app over a list of configs and collect tag results."""
    all_results = []
    for config in tqdm(configs, desc="Processing configs"):
        yaml_content = config["file_content"]
        file_tags = []

        # Renamed 'missconfig' to 'summary' for better clarity.
        for summary in summarize_yaml(yaml_content, llm):
            initial_state = {
                "messages": ["start"],
                "yaml": yaml_content,
                "summary": summary,
                "tags": [],
            }
            # Using more descriptive names for the state before and after invocation.
            final_state = app.invoke(initial_state)
            if final_state.get("tags"):
                file_tags.extend(final_state["tags"])

        all_results.append({"file_name": config["file"], "tags": file_tags})
        print(f"Processed {config['file']} with tags: {file_tags}\n\n")

    return all_results


def save_results(results: list, out_dir: str = "results", tags_tool: str = "checkov"):
    """Persist results to a timestamped JSON file and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(out_dir, f"agent_results_tags_{tags_tool}_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return output_path


def _setup_agent_and_tools(tags_tool: str, with_validation: bool, model_provider: str, model_name: str):
    """A helper function to handle all the setup logic for the agent."""
    if not os.path.exists(f"rag_db_{tags_tool}"):
        start_rag(tags_tool)
    else:
        print("RAG DB already initialized. Skipping setup.")

    llm = setup_llm(model_provider, model_name)
    rag_tool = build_rag_tool(tags_tool)
    tools = [search_tool, rag_tool]
    llm_with_tools = llm.bind_tools(tools=tools)

    validator_config = None
    if with_validation:
        with open("misconfigs_map.json", "r", encoding="utf-8") as f:
            all_tag_defs = json.load(f)
        tag_definitions = all_tag_defs.get(tags_tool, {})
        validator_config = {"llm": llm, "tag_definitions": tag_definitions}

    app = build_graph(llm_with_tools, tools, validator_config)
    return app, llm


def run_agent_for_tool(filename: str, tags_tool: str, limit: int, with_validation: bool, model_provider: str, model_name: str):
    """Run the agent on a single tool's dataset and evaluate the results."""
    app, llm = _setup_agent_and_tools(tags_tool, with_validation, model_provider, model_name)

    with open(filename, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if limit > 0:
        configs = configs[:limit]

    results = process_configs(app, configs, llm)
    result_path = save_results(results, tags_tool=tags_tool)

    # Evaluate results
    print(f"\n--- Evaluating results for {tags_tool} ---")
    print("Evaluating without normalization:")
    evaluate_llm_per_tool(result_path, tools_to_compare=[tags_tool, tags_tool + "_new"])
    print("\nEvaluating with normalization:")
    evaluate_llm_per_tool_with_normalize(result_path, tools_to_compare=[tags_tool, tags_tool + "_new"])


def main():
    """Main entry point for running the agent, with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Run the KubeSecAgent to identify misconfigurations.")
    parser.add_argument(
        "--filename", type=str, default="10k_open_source_manifests.json",
        help="Path to the JSON file containing Kubernetes manifests."
    )
    parser.add_argument(
        "--tags-tools", nargs='+', default=["checkov"],
        help='List of tools to run (e.g., "checkov", "kube_linter").'
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Maximum number of configs to process per tool. -1 for all."
    )
    parser.add_argument(
        "--model-provider", type=str, default="openai", choices=["openai", "google", "huggingface"],
        help="The provider for the language model."
    )
    parser.add_argument(
        "--model-name", type=str, default="gpt-4o",
        help='Model to use. E.g., "gpt-4o", "gemini-1.5-pro", or a Hugging Face ID like "NousResearch/Hermes-2-Pro-Llama-3-8B".'
    )
    parser.add_argument(
        "--no-validation", action="store_false", dest="with_validation",
        help="Disable the result validation node in the graph."
    )
    args = parser.parse_args()

    for tool in tqdm(args.tags_tools, desc="Running agents for tools"):
        try:
            print(f"\n===== Running agent for tool: {tool} using model: {args.model_name} =====")
            run_agent_for_tool(
                filename=args.filename,
                tags_tool=tool,
                limit=args.limit,
                with_validation=args.with_validation,
                model_provider=args.model_provider,
                model_name=args.model_name
            )
            print(f"===== Finished processing with tool: {tool} =====")
        except Exception as e:
            # raise e
            print(f"Error processing tool {tool}: {e}")
            continue


if __name__ == "__main__":
    # Example command-line usages:
    #
    # Run with default OpenAI GPT-4o on 5 manifests:
    # python main.py --limit 5
    #
    # Run with Google Gemini:
    # python main.py --model-provider google --model-name gemini-1.5-pro-latest --limit 5
    #
    # Run with a local Hugging Face model (ensure it's fine-tuned for tool-calling):
    # python main.py --model-provider huggingface --model-name NousResearch/Hermes-2-Pro-Llama-3-8B --limit 5 --no-validation
    main()