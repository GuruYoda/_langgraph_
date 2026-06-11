import sys
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama

# 1. Define the Graph State
# This track and appends conversation history using 'add_messages'
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Initialize the local DeepSeek R1 8B Model via Ollama
# Adjust temperature if you want more structured (lower) or creative (higher) reasoning
llm = ChatOllama(
    model="llama3.1", 
    temperature=1.0
)

# 3. Define the Chatbot Node
def chatbot_node(state: State):
    # Sends the current thread of messages to the model
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 4. Construct the Graph Topology
graph_builder = StateGraph(State)

# Add our processing node to the graph
graph_builder.add_node("chatbot", chatbot_node)

# Define execution flow (Start -> Chatbot -> End)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the workflow into an executable graph
graph = graph_builder.compile()

# 5. Interactive Chat Loop Execution
if __name__ == "__main__":
    print("--- LangGraph Chatbot Initialized (Type 'exit' or 'quit' to stop) ---")
    
    # Initialize an empty list to keep track of memory across loops
    current_messages = []
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
                
            if not user_input.strip():
                continue

            # Append the new user message to the state
            current_messages.append(("user", user_input))
            
            # Run the graph and stream the state updates
            # stream_mode="values" outputs the complete state at each step
            events = graph.stream({"messages": current_messages}, stream_mode="values")
            
            # Get the latest state update and extract the model's response
            final_event = None
            for event in events:
                final_event = event
            
            if final_event and "messages" in final_event:
                # Get the last message added (the AI's reply)
                ai_message = final_event["messages"][-1]
                print(f"\nAI:\n{ai_message.content}")
                
                # Keep the persistent memory updated
                current_messages = final_event["messages"]
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)