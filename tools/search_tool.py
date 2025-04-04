from typing import Annotated
from langgraph.types import Command, interrupt
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_community.tools.tavily_search import TavilySearchResults

@tool
def human_assistance(tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Tool interruption for human assistance and verification
    """
    human_response = interrupt({"query": 'Is this correct?'})

    if human_response.get("correct", "").lower().startswith("y"):
        response = 'Correct'
    else:
        response = f"Made a correction: {human_response}"

    # Update the state ToolMessage explicitly
    state_update = {
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)]
    }
    return Command(update=state_update)

search_tool = TavilySearchResults(max_result=2) 
tools = [search_tool, human_assistance]