import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from ast import literal_eval

LABELS_PATH = "10k_open_source_labels.csv"


def evaluate_llm_per_tool(llm_json_path, tools_csv_path=LABELS_PATH, tools_to_compare=None):
    """
    Evaluate LLM-detected tags against specific tool-detected tags.

    Args:
        llm_json_path (str): Path to the LLM output JSON.
        tools_csv_path (str): Path to the tools output CSV.
        tools_to_compare (list[str] or None): Subset of tools to evaluate against,
            e.g., ['checkov'] or ['kube_linter', 'terrascan'].
            If None, all tools are used.
    """

    # Load LLM results
    with open(llm_json_path, "r", encoding="utf-8") as f:
        llm_data = json.load(f)
    llm_df = pd.DataFrame(llm_data).rename(columns={"file_name": "file"})

    # Filter out entries with no meaningful LLM tags
    llm_df["tags"] = llm_df["tags"].apply(lambda x: [t for t in x if t != "no_error"])
    llm_df = llm_df[llm_df["tags"].map(len) > 0]

    # Load tool-based results
    tools_df = pd.read_csv(tools_csv_path)
    tools_df['file'] = tools_df['file'].astype(str)
    llm_df['file'] = llm_df['file'].astype(str)

    # Merge LLM and tools
    merged = pd.merge(llm_df, tools_df, on="file", how="inner")

    # All possible tools
    tool_columns = {
        "kube_linter": "kube_linter_eids",
        "kube_linter_new": "kube_linter_new_eids",
        "terrascan": "terrascan_eids",
        "terrascan_new": "terrascan_new_eids",
        "checkov": "checkov_eids",
        "checkov_new": "checkov_new_eids",

    }

    # Filter tools if a subset is specified
    if tools_to_compare is not None:
        tool_columns = {k: v for k, v in tool_columns.items() if k in tools_to_compare}

    results = {}

    for tool_name, tool_col in tool_columns.items():
        precisions, recalls, f1s = [], [], []

        for _, row in merged.iterrows():
            try:
                llm_tags = set(row["tags"])

                # Clean and evaluate escaped list strings safely
                tool_raw = row[tool_col]
                if isinstance(tool_raw, str):
                    tool_clean = tool_raw.replace("\\'", "'").replace('\"', '"')
                    tool_tags = set(literal_eval(tool_clean))
                else:
                    tool_tags = set()

            except Exception:
                continue

            # Remove "no_error" tags
            llm_tags.discard("no_error")
            tool_tags.discard("no_error")

            all_tags = llm_tags.union(tool_tags)
            if not all_tags:
                continue

            y_true = [1 if tag in tool_tags else 0 for tag in all_tags]
            y_pred = [1 if tag in llm_tags else 0 for tag in all_tags]

            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1s.append(f1_score(y_true, y_pred, zero_division=0))

        results[tool_name] = {
            "precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "f1": sum(f1s) / len(f1s) if f1s else 0.0
        }

    # Print results
    for tool, metrics in results.items():
        print(f"\n🔍 LLM Agent vs {tool.capitalize()}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")

    return results
