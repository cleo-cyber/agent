from typing import Literal
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langgraph.graph import END,START,StateGraph
from typing import TypedDict, Sequence, Annotated
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class AgentState(TypedDict):
    # Append messages 
    messages: Annotated[Sequence[BaseMessage],add_messages]


class  AgentBuilder:
    def __init__(self,llm,retriever_tool):
        self.llm=llm
        self.retriever_tool=retriever_tool
        # self.memory=MemorySaver()
    
    def agent(self,state)->dict:
        """
        The main agent function that handles the decision-making process.
        Invokes the llm to generate a response based on current state.
        Given a question, it decides to retrieve using retriever tool or simply generate a response.

        Args:
            state (AgentState - messages): The current state of the agent.
        Returns:
            dict: The updated state with the agent response appended to the messages.
        """

        print("Agent invoked")
        tools=self.retriever_tool
        messages=state['messages']
        model=self.llm.bind_tools([tools])
        response=model.invoke(messages)

        return {"messages":[response]}
    
    def grade_documents(self,state) -> Literal["generate","rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state
        Returns:
            str: A decision, whether the documents are relevant or not.

        """

        class grade(BaseModel):
            """
            Score for relevance check

            """

            binary_score: str=Field(description="Relevance score 'yes' or 'no")


        llm_with_tool=self.llm.with_structured_output(grade)

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        chain=prompt | llm_with_tool

        messages=state['messages']
        last_message=messages[-1]
        question=messages[0].content
        docs=last_message.content

        scored_result=chain.invoke({"question":question,"context":docs})

        score=scored_result.binary_score

        if score=="yes":
            print("-- DECISION: Relevant --")
            return "generate"
        else:
            print("-- DECISION: Not Relevant --")
            print("scoring: ",score)
            print("rewriting the document")
            return "rewrite"
        
    def rewrite(self,state):
        """
        Transform the query to produce better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with rephrased question
        """

        print("Rewriting the question")
        messages=state['messages']
        question=messages[0].content

        msg=[HumanMessage(
            content=f""" \n 
                Look at the input and try to reason about the underlying semantic intent / meaning. \n 
                Here is the initial question:
                \n ------- \n
                {question} 
                \n ------- \n
                Formulate an improved question: """,       
            )]
        
        response=self.llm.invoke(msg) 
        return {"messages":[response]}

    def generate(self,state)->dict:
        """
        Generates a response based on the current state.

        Args:
            state (messages): The current state
            
        returns:
            dict: The updated state with the generated response appended to the messages.

        """

        print("Generating the response")

        messages=state['messages']
        question=messages[0].content
        last_message=messages[-1]

        docs=last_message.content

        # prompt
        prompt=PromptTemplate(
            template="""
            You are an assistant that is given a question and a retrieved document. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            Answer the question based on the retrieved document. \n
            """,
            input_variables=["context", "question"],
        )

        rag_chain=prompt | self.llm | StrOutputParser()

        response=rag_chain.invoke({"context":docs,"question":question})

        print("Response: ",response)

        return {"messages":[response]}

    def create_graph(self)->StateGraph:
        """
        Create the state graph for the agent.
        Args:
            retriever_tool (ToolNode): The retriever tool node.
        Returns:

            StateGraph: The compiled state graph.
        """
        graph=StateGraph(AgentState)
        graph.add_node('agent',self.agent)
        retriever=ToolNode([self.retriever_tool])
        graph.add_node('retriever',retriever)
        graph.add_node('rewrite',self.rewrite)
        graph.add_node('generate',self.generate)

        self._add_edge(graph)

        return graph.compile()
    
    def _add_edge(self,graph):
        """
        Add edges to the state graph.
        Args:
            graph (StateGraph): The state graph to which edges will be added.
        """
        graph.add_edge(START,'agent')
        graph.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools":"retriever",
                END:END
                }
        )

        graph.add_conditional_edges(
            'retriever',
            self.grade_documents)
        graph.add_edge("generate",END)
        graph.add_edge("rewrite",END)




