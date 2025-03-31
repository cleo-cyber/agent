import os 
from typing import Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.types import Command, interrupt
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import InjectedToolCallId, tool
from langchain.tools.tavily_search import TavilySearchResults

load_dotenv()

os.environ['LANGSMITH_ENDPOINT']=os.getenv('LANGSMITH_ENDPOINT')
os.environ['LANGSMITH_API_KEY']=os.getenv('LANGSMITH_API_KEY')
os.environ["LANGSMITH_PROJECT"] = os.getenv('LANGSMITH_PROJECT')
os.environ['GQOQ_API_KEY']=os.getenv('GQOQ_API_KEY')
os.environ['TAVILY_API_KEY']=os.getenv('TAVILY_API_KEY')
GROQ_API_KEY=os.getenv('GQOQ_API_KEY')
MODEL_NAME=os.getenv('MODEL_NAME')

llm=ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

memory=MemorySaver()

@tool
def human_assistance(query:str,tool_call_id: Annotated[str,InjectedToolCallId]):
    """
    Tool interruption for human assistance and verification
    """
    human_response=interrupt({"query":'Is this correct?'})

    if human_response.get("correct","").lower().startswith("y"):
        response='Correct'
    else:
        response=f"Made a correction: {human_response}"
    
    # Update the state ToolMessage explicitly
    state_update={
        "messages":[ToolMessage(response,tool_call_id=tool_call_id)]
    }
    return Command(update=state_update)

tool=TavilySearchResults(max_result=2)
tools=[tool,human_assistance]
llm_with_tools=llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list,add_messages]

def chatbot(state:State):
    message= llm_with_tools.invoke(state["messages"])

    assert len(message.tool_calls)<=1
    return {"messages":[message]}



def build_graph():
    graph_builder=StateGraph(State)
    tool_node=ToolNode(tools=[tool,human_assistance])

    graph_builder.add_node('chatbot',chatbot)
    graph_builder.add_node('tools',tool_node)

    graph_builder.add_conditional_edges('chatbot',tools_condition)
    graph_builder.add_edge('tools','chatbot')
    graph_builder.add_edge(START,'chatbot')

    return graph_builder.compile(checkpointer=memory)

def stream_graph_updates(user_input:str):
    graph = build_graph()
    try:
        events = graph.stream(
            {"messages":[{"role":"user","content":user_input}]},
            {"configurable": {"thread_id": "1"}},
            stream_mode="values"
        )
        
        for event in events:
            try:
                event["messages"][-1].pretty_print()
            except UnicodeEncodeError:
                content = event['messages'][-1].content
                safe_content = content.encode(encoding='utf-8',errors='replace').decode('utf-8')
                print(f"Assistant: {safe_content}")
            except Exception as e:
                print(f"Error processing event: {e}")
    except GeneratorExit:
        print("Stream was closed unexpectedly")
    except Exception as e:
        print(f"Error in streaming: {e}")

if __name__== "__main__":
    while True:
        try:
            user_input=input('User: ')
            if user_input.lower() in ['quit','exit','q']:
                print('Goodbye')
                break
            stream_graph_updates(user_input)
        except Exception as e:
            user_input='Failed to process'
            print(e)

            break

    
