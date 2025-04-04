from config.config import Config
from langchain_groq import ChatGroq
from tools.search_tool import tools
from langgraph.checkpoint.memory import MemorySaver
from graph.search_agent_builder import GraphBuilder
from handlers.search_stream_handler import StreamHandler

def main():
    config=Config()
    llm=ChatGroq(groq_api_key=config.GROQ_API_KEY, model_name=config.MODEL_NAME)
    memory=MemorySaver()

    graph_builder=GraphBuilder(llm=llm,tools=tools,memory=memory)
    graph=graph_builder.build_graph()

    while True:
        user_input=input("User: ")
        if user_input.lower() in ["exit","quit"]:
            break
        try:
            StreamHandler.handle_stream(graph,user_input)
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__=="__main__":
    main()