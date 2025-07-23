import re

from langgraph.prebuilt import ToolNode
from langgraph.graph import END , add_messages
from typing import TypedDict, Annotated
# from tools import tools
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import HumanMessage, AIMessage

# define intial state 
# optional makes the field able to hold none 
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # user/agent conversation
    yaml: Optional[str] 
    summary: Optional[str]  
    tags : Optional[list[str]]


def k_expert(llm):
    """Return a stateful expert function for a LangGraph node.

    The returned ``expert`` callable expects a ``state`` mapping compatible with
    :class:`AgentState` and updates it based on LLM output. ``state`` should
    contain at least the following keys:

    - ``messages``: list of prior :class:`HumanMessage`/``AIMessage`` objects.
    - ``yaml``: the Kubernetes manifest under analysis (``str`` or ``None``).
    - ``summary``: a one-line description of the misconfiguration.
    - ``tags``: list of already selected misconfiguration tags.

    On each invocation the LLM receives the current conversation, YAML and
    summary, then replies with additional guidance. The assistant message is
    appended to ``messages``. If the reply includes ``finish``, any tags
    following ``tags:`` are parsed and merged into ``state['tags']`` before the
    updated ``state`` is returned.
    """
    def expert(state):
        """Run a single reasoning step with the LLM and update ``state``.

        Parameters
        ----------
        state : AgentState
            Current conversation state passed between graph nodes.

        Returns
        -------
        dict
            Updated state with any new assistant message and extracted tags.
        """
        messages = state.get("messages", [])
        yaml = state.get("yaml", [])
        summary = state.get("summary")
        prompt = f"""
You are a Kubernetes security expert.

Your task:
1. Analyze the following Kubernetes YAML and misconfig summary sentence.

** It's important to know you only use the rag tool once, then you must continue and choose the most relevant tag 
Instructions:
Do not invent or add extra problems.
You may only use 1 sentence per tool call.

This is the context massages:
{
    messages
}
The Rag tool retrieve  options . you need to choose the most relevant one for the summary 

Yaml:
{yaml}

Input summary:
{summary}

Look at previous AI messages, if the rag tool was already usd do not use it again.
you need to choose the most relevant tag for the summary 

Expected final response format **exactly**, if none match, answer `no_error`:

** finish **
tags: <tag-1>, <tag-2>, …, <tag-N>
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        print(response.content)

        # Always append the assistant message to history
        new_state = {"messages": messages + [response]}

        if "finish" in response.content.lower():
            # Extract everything after the first ':' on the 'tags:' line
            tag_line = response.content.split(":", 1)[1]
            # Split on commas or new-lines, strip whitespace, keep non-empty
            new_tags = [t.strip() for t in re.split(r"[,\n]", tag_line) if t.strip()]

            # Merge with any tags we already have
            merged_tags = list(dict.fromkeys(state.get("tags", []) + new_tags))  # dedupe, preserve order
            new_state["tags"] = merged_tags

        return new_state
        # if "finish" in response.content:
        #             return {
        #     "messages": [response],
        #     "tag": response.content.split(":")[1].strip()
        # }
        # else:
        #     return {
        #         "messages": [response],
        #     }

    return expert


# conditional edge
def tool_use(state: AgentState):
    """Return the next node name if the conversation requests a tool.

    Parameters
    ----------
    state : AgentState
        Current conversation state.

    Returns
    -------
    str
        ``"tools"`` if a tool call is present, otherwise ``END``.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    # print tool call
    if tool_calls:
        print("Tool request:")
        for tool_call in tool_calls:
            print(f"Tool name: {tool_call['name']}")
        #  print(f"Arguments: {tool_call['args']}")
                
        return "tools"
    else:
        return END


class TagValidator:
    """
    Filter out false positives produced by k_expert.
    Keeps only tags explicitly confirmed by an LLM (or rule engine).
    """
    def __init__(self, llm, tag_definitions):
        """Create a new validator instance.

        Parameters
        ----------
        llm : Any
            Chat model used to confirm misconfigurations.
        tag_definitions : dict[str, str]
            Mapping of tag names to canonical rule descriptions.
        """
        self.llm = llm
        self.tag_defs = tag_definitions

    def __call__(self, state):
        """Validate detected tags by asking the LLM for confirmation.

        Parameters
        ----------
        state : AgentState
            Current conversation state containing ``yaml``, ``summary``
            and detected ``tags``.

        Returns
        -------
        AgentState
            The updated state with only confirmed tags retained.
        """
        yaml_doc  = state["yaml"]
        summary   = state["summary"]
        tags      = state.get("tags", [])

        confirmed = []
        for tag in tags:
            tag_def = self.tag_defs.get(tag, "")
            prompt = f"""You are a Kubernetes security auditor.\n
YAML:\n{yaml_doc}\n
Issue summary:\n{summary}\n
Proposed tag: {tag}\nDefinition: {tag_def}\n
Question: Does the YAML truly violate this rule? Reply with 'YES' or 'NO' and one-line reason."""
            # reply = self.llm.invoke(prompt).content.strip().lower()
            reply = self.llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
            print(f"Tag: {tag}, Reply: {reply}")
            if reply.startswith("yes"):
                confirmed.append(tag)

        state["tags"] = confirmed
        return state

