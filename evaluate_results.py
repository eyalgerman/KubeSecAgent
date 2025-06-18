import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from ast import literal_eval
import json
LABELS_PATH = "10k_open_source_labels.csv"
MISS_CONFIG_MAP_PATH = "misconfigs_map.json"


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


def build_normalization_map_from_json(json_path, tool_name):
    """
    Loads missconfig_map.json and returns a normalization mapping for a given tool.
    Each tag (LLM or tool) with the same description is mapped to the first tag found with that description.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        missconfig_map = json.load(f)

    tool_map = missconfig_map.get(tool_name, {})  # e.g., "kube_linter"
    if not tool_map:
        tool_map = missconfig_map.get(tool_name.split("_")[0], {})
    desc_to_tag = {}
    tag_norm_map = {}

    for tag, desc in tool_map.items():
        # Assign first tag as canonical for a given description
        if desc not in desc_to_tag:
            desc_to_tag[desc] = tag
        tag_norm_map[tag] = desc_to_tag[desc]

    return tag_norm_map

def normalize_tags(tags, tag_norm_map):
    return set(tag_norm_map.get(tag, tag) for tag in tags)



def evaluate_llm_per_tool_with_normalize(
    llm_json_path,
    tools_csv_path=LABELS_PATH,
    missconfig_json_path=MISS_CONFIG_MAP_PATH,
    tools_to_compare=None
):
    """
    Evaluate LLM-detected tags against specific tool-detected tags,
    normalizing all tags using missconfig_map.json for a fair comparison.
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

    # Tool-specific columns
    tool_columns = {
        "kube_linter": "kube_linter_eids",
        "kube_linter_new": "kube_linter_new_eids",
        "terrascan": "terrascan_eids",
        "terrascan_new": "terrascan_new_eids",
        "checkov": "checkov_eids",
        "checkov_new": "checkov_new_eids",
    }

    if tools_to_compare is not None:
        tool_columns = {k: v for k, v in tool_columns.items() if k in tools_to_compare}

    results = {}

    for tool_shortname, tool_col in tool_columns.items():
        tag_norm_map = build_normalization_map_from_json(missconfig_json_path, tool_shortname)
        precisions, recalls, f1s = [], [], []
        skipped = 0

        if tool_col not in merged.columns:
            print(f"Warning: Tool column '{tool_col}' not found in data. Skipping.")
            continue

        for _, row in merged.iterrows():
            try:
                llm_tags = set(row["tags"])
                tool_raw = row[tool_col]
                if isinstance(tool_raw, str):
                    tool_clean = tool_raw.replace("\\'", "'").replace('\"', '"')
                    tool_tags = set(literal_eval(tool_clean))
                else:
                    tool_tags = set()
            except Exception:
                skipped += 1
                continue

            llm_tags.discard("no_error")
            tool_tags.discard("no_error")
            llm_tags = normalize_tags(llm_tags, tag_norm_map)
            tool_tags = normalize_tags(tool_tags, tag_norm_map)

            all_tags = llm_tags.union(tool_tags)
            if not all_tags:
                skipped += 1
                continue

            y_true = [1 if tag in tool_tags else 0 for tag in all_tags]
            y_pred = [1 if tag in llm_tags else 0 for tag in all_tags]

            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1s.append(f1_score(y_true, y_pred, zero_division=0))

        results[tool_shortname] = {
            "precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "skipped": skipped,
            "count": len(precisions)
        }

    for tool, metrics in results.items():
        print(f"\n🔍 LLM Agent vs {tool.capitalize()}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  Skipped:   {metrics['skipped']} (out of {metrics['skipped'] + metrics['count']})")

    return results

if __name__ == "__main__":
    # Example usage
    llm_json_path = "results/agent_results_tags_checkov_20250617_115228.json"
    tool_name = "checkov"
    tools_csv_path = LABELS_PATH
    missconfig_json_path = MISS_CONFIG_MAP_PATH

    # Evaluate without normalization
    print("Evaluating without normalization:")
    evaluate_llm_per_tool(llm_json_path, tools_csv_path, tools_to_compare=[tool_name])

    # Evaluate with normalization for
    print(f"\nEvaluating with normalization for {tool_name}:")
    evaluate_llm_per_tool_with_normalize(
        llm_json_path, tools_csv_path, missconfig_json_path, tools_to_compare=[tool_name]
    )

