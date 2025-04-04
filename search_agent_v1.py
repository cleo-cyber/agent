import os
from typing import Annotated
from tools.tools import BasicToolNode
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

os.environ['LANGSMITH_ENDPOINT']=os.getenv('LANGSMITH_ENDPOINT')
os.environ['LANGSMITH_API_KEY']=os.getenv('LANGSMITH_API_KEY')
os.environ["LANGSMITH_PROJECT"] = os.getenv('LANGSMITH_PROJECT')
os.environ['GQOQ_API_KEY']=os.getenv('GQOQ_API_KEY')
os.environ['TAVILY_API_KEY']=os.getenv('TAVILY_API_KEY')

GROQ_API_KEY=os.getenv('GQOQ_API_KEY')

llm=ChatGroq(groq_api_key=GROQ_API_KEY,model_name='Gemma2-9b-It')

    
tool=TavilySearchResults(max_results=2)
tools=[tool]
llm_with_tools=llm.bind_tools(tools)

class State(TypedDict):
    messages:Annotated[list,add_messages]

def set_tools():
    tool=TavilySearchResults(max_results=2)
    tools=[tool]
    return tools

def chatbot(state: State):
    return {'messages':[llm_with_tools.invoke(state['messages'])]}

def route_tools(state:State):

    if isinstance(state,list):
        ai_message=state[-1]

    elif message :=state.get("messages",[]):
        ai_message=message[-1]

    else:
        raise ValueError(f'No messages found in input state to tool_edge: {state}')
    
    if hasattr(ai_message,"tool_calls") and len(ai_message.tool_calls)>0:
        return 'tools'
    
    return END

def build_graph():
    graph_builder=StateGraph(State)
    tool_node=BasicToolNode(tools=[tool])

    graph_builder.add_node('tools',tool_node)
    graph_builder.add_node('chatbot',chatbot)
    graph_builder.add_conditional_edges('chatbot',route_tools,{"tools":"tools",END:END},)
    graph_builder.add_edge('tools','chatbot')
    graph_builder.add_edge(START,'chatbot')
    # graph_builder.add_edge('chatbot',END)

    return graph_builder.compile()


def stream_graph_updates(user_input:str):
    graph=build_graph()
    for event in graph.stream({"messages":[{"role":"user","content":user_input}]}):
        for value in event.values():
            print('Assistant: ',value['messages'][-1].content)

if __name__ == "__main__":
    while True:
        try:
            user_input=input('User: ')
            if user_input.lower() in ['quit','exit','q']:
                print('Goodbye')
                break
            stream_graph_updates(user_input)
        except:
            user_input='What do you know about LangGraph'
            print('User '+ user_input)

            stream_graph_updates(user_input)
            break



