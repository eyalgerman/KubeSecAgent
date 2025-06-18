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
    tag : Optional[str]     


def k_expert(llm):
    def expert(state):
        messages = state.get("messages", [])
        yaml = state.get("yaml", [])
        summary = state.get("summary")
        prompt = f"""
You are a Kubernetes security expert.

Your task:
1. Analyze the following Kubernetes YAML and misconfig summary sentence.

** its important to know you only use the rag tool once . then you must continue and choose the most relevant tag 
Instructions:
Do not invent or add extra problems.
You may only use 1 sentence per tool call.

this is the context massages:
{
    messages
}
The Rag tool retrive  options . you need to choose the most relevant one for the summary 



yaml:
{yaml}

Input summary:
{summary}

look at previous AI messages . if th rag tool was already usd do not use it again.
you need to choose the most relevant tag  for the summary 

Expected final response format:

** finish **
the most relevant tag is :
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        print(response.content)
        if "finish" in response.content:
                    return {
            "messages": [response],
            "tag": response.content.split(":")[1].strip()
        }
        else:
            return {
                "messages": [response],
            }

    return expert


# conditional edge
def tool_use(state: AgentState):
    """Check if the last message contains tool calls."""
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
    

