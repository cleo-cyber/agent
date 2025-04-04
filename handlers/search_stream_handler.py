class StreamHandler:
    @staticmethod
    def handle_stream(graph, user_input)-> None:
        """
        Handle the streaming of events from the graph.
        Args:
            graph (StateGraph): The state graph to stream from.
            user_input (str): The user input to process.
        """
        
        try:
            events= graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                {"configurable": {"thread_id": "1"}},
                stream_mode="values"
            )

            for event in events:
                try:
                    if event.tool_calls:
                        for tool_call in event.tool_calls:
                            print(f"Tool Call: {tool_call}")
                    else:
                        print(f"Message: {event.messages[-1].content}")
                except Exception as e:
                    print(f"Error processing event: {e}")
        except GeneratorExit:
            print("Stream was closed unexpectedly")
        except Exception as e:
            print(f"Error in streaming: {e}")
