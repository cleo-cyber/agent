from typing import Annotated
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition

class State(TypedDict):
    messages: Annotated[list,add_messages]

class GraphBuilder:
    def __init__(self,llm:ChatGroq,tools:list,memory):
        """
        Initialize the GraphBuilder with the LLM, tools, and memory.
        Args:
            llm (ChatGroq): The LLM to be used.
            tools (list): List of tools to be used in the agent.
            memory: Memory object for state management.
        """

        self.llm=llm
        self.tools=tools
        self.memory=memory
        self.llm_with_tools=self.llm.bind_tools(self.tools)

    def chatbot(self,state:State):
        
        message= self.llm_with_tools.invoke(state["messages"])
        print(f"Message type: {type(message)}")
        print(f"Message content: {message}")

        assert len(message.tool_calls)<=1
        return {"messages":[message]}
    
    def build_graph(self)->StateGraph:
        """
        Build the state graph for the agent.
        This graph defines the flow of the agent's decision-making process.
        """
        graph_builder=StateGraph(State)
        tool_node=ToolNode(self.tools)

        graph_builder.add_node('chatbot',self.chatbot)
        graph_builder.add_node('tools',tool_node)

        self._add_edges(graph_builder)
        return graph_builder.compile(checkpointer=self.memory)
    
    def _add_edges(self,graph_builder:StateGraph):
        """
        Add edges to the graph.
        Args:
            graph_builder (StateGraph): The state graph to which edges will be added.

        """
        graph_builder.add_conditional_edges('chatbot',tools_condition)
        graph_builder.add_edge('tools','chatbot')
        graph_builder.add_edge(START,'chatbot')

    