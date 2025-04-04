from config.config import Config
from orchestration.orchestrator import RAGOrchestration

def main():
    print("Starting RAG Orchestration")
    
    config=Config()

    orchestration=RAGOrchestration(config)

    graph=orchestration.run()

    print("RAG Orchestration completed")

    query=input("Enter your query: ")

    print("Running the query")
    input_message={
        "messages":[
            ("user",str(query))
        ]
    }

    for outputs in graph.stream(input_message):
        for k,v in outputs.items():
            print(f"Outputs from node: {k}")
            print("------------------------------------------------------")
            print(v)
            print("\n")


if __name__=="__main__":
        main()