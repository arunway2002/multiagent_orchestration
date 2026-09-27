import asyncio
from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
# 1. Initialize the Local Llama 3.2 model via Ollama
# By default, it connects to http://localhost:11434
llm = ChatOllama(
    model="llama3.2",  # Use "llama3.2" for the 3B version if your Mac has 16GB+ RAM
    temperature=0
)

# 2. Define the shared state dictionary that flows between nodes
class AgentState(TypedDict):
    customer_message: str
    category: str
    final_reply: str

# 3. Define Node 1: Intent Classification
def classify_intent_node(state: AgentState):
    print("--- [Node 1] Classifying Customer Intent ---")
    message = state["customer_message"]
    
    prompt = (
        f"Analyze this customer message: '{message}'. "
        "Classify it into exactly one category from: [Technical, Billing, General]. "
        "Respond with ONLY the category word."
    )
    
    response = llm.invoke(prompt)
    category_out = response.content.strip()
    
    return {"category": category_out}

# 4. Define Node 2: Tailored Response Generation
def generate_response_node(state: AgentState):
    print("--- [Node 2] Generating Custom Response ---")
    category = state["category"]
    message = state["customer_message"]
    
    prompt = (
        f"Write a short 1-sentence customer support response. "
        f"The customer's problem is: '{message}'. "
        f"The department handling this is: '{category}'."
    )
    
    response = llm.invoke(prompt)
    return {"final_reply": response.content.strip()}

# 5. Build the State Workflow Graph
workflow = StateGraph(AgentState)

# Add our custom processing nodes to the graph
workflow.add_node("classifier", classify_intent_node)
workflow.add_node("responder", generate_response_node)

# Set up the sequential logic paths (Edges)
workflow.add_edge(START, "classifier")       # Start here
workflow.add_edge("classifier", "responder")  # Move from node 1 to node 2
workflow.add_edge("responder", END)           # End the workflow

# Compile the graph architecture
app = workflow.compile()

# --- 6. Execute the Graph Workflow ---
async def main():
    initial_input = {
        "customer_message": "My screen goes completely black whenever I try to open the checkout page."
    }
    
    print("Starting Orchestrated Workflow...\n")
    
    # Track the final state data manually as it streams through
    final_output = {}
    async for output in app.astream(initial_input):
        for node_name, state_update in output.items():
            print(f"Incremental Update from [{node_name}]: {state_update}\n")
            final_output.update(state_update)

    print("=" * 40)
    print("🎉 FINAL ORCHESTRATION OUTPUT:")
    print(f"Inferred Category: {final_output.get('category')}")
    print(f"Generated Response: {final_output.get('final_reply')}")
    print("=" * 40)

if __name__ == "__main__":
    asyncio.run(main())