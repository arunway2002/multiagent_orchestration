import asyncio
import os
from typing import TypedDict, Literal
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. SETUP MODEL AND VECTOR DATABASES
# =====================================================================
print("--- Initializing Local Vector Databases for Sub-Agents ---")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:latest", temperature=0)

# Helper function using langchain_community TextLoader to build isolated stores
def build_vector_store(file_path: str) -> InMemoryVectorStore:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Please create the file '{file_path}' before running.")
        
    # Use LangChain Community TextLoader
    loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = text_splitter.split_documents(documents)
    
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)
    return store

# Build 3 separate isolated knowledge bases from your files
journal_store = build_vector_store("finance_journal.txt")
tax_store = build_vector_store("tax_validation.txt")
general_store = build_vector_store("corporate_general.txt")

# =====================================================================
# 2. STATE DEFINITIONS
# =====================================================================
class AgentState(TypedDict):
    question: str
    next_agent: Literal["journal_agent", "tax_agent", "general_agent"]
    retrieved_context: str
    final_answer: str

# =====================================================================
# 3. ROUTER / SUPERVISOR NODE
# =====================================================================
def supervisor_router_node(state: AgentState):
    print("\n[Supervisor] Analyzing Intent to Route Context...")
    question = state["question"].lower()
    
    # Semantic routing heuristics based on user query keywords
    if any(word in question for word in ["journal", "entry", "account", "debit", "credit", "je-"]):
        next_step = "journal_agent"
    elif any(word in question for word in ["tax", "invoice", "vat", "deduction", "withholding"]):
        next_step = "tax_agent"
    else:
        next_step = "general_agent"
        
    print(f"-> Routing to: {next_step}")
    return {"next_agent": next_step}

# =====================================================================
# 4. SUB-AGENT NODES (RAG WORKFLOWS)
# =====================================================================
def journal_agent_node(state: AgentState):
    print("\n[Journal Agent] Searching accounting entries ledger...")
    docs = journal_store.as_retriever(search_kwargs={"k": 1}).invoke(state["question"])
    context = docs[0].page_content if docs else "No account data found."
    
    prompt = f"You are a Finance Journal Entry expert. Answer using this context:\n{context}\n\nQ: {state['question']}\nA:"
    res = llm.invoke(prompt)
    return {"final_answer": res.content.strip(), "retrieved_context": context}

def tax_agent_node(state: AgentState):
    print("\n[Tax Agent] Analyzing tax statutes & validations...")
    docs = tax_store.as_retriever(search_kwargs={"k": 1}).invoke(state["question"])
    context = docs[0].page_content if docs else "No tax matrix matching data found."
    
    prompt = f"You are a Corporate Tax Auditor. Answer using this context:\n{context}\n\nQ: {state['question']}\nA:"
    res = llm.invoke(prompt)
    return {"final_answer": res.content.strip(), "retrieved_context": context}

def general_agent_node(state: AgentState):
    print("\n[General Operations Agent] Checking company handbook guidelines...")
    docs = general_store.as_retriever(search_kwargs={"k": 1}).invoke(state["question"])
    context = docs[0].page_content if docs else "No company context found."
    
    prompt = f"You are a general operations assistant. Answer using this context:\n{context}\n\nQ: {state['question']}\nA:"
    res = llm.invoke(prompt)
    return {"final_answer": res.content.strip(), "retrieved_context": context}

# =====================================================================
# 5. GRAPH BUILDING & CONDITIONAL ROUTING LOGIC
# =====================================================================
workflow = StateGraph(AgentState)

# Add processing nodes
workflow.add_node("supervisor", supervisor_router_node)
workflow.add_node("journal_agent", journal_agent_node)
workflow.add_node("tax_agent", tax_agent_node)
workflow.add_node("general_agent", general_agent_node)

# Set Graph Entry Point
workflow.set_entry_point("supervisor")

# Define conditional logic map link
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "journal_agent": "journal_agent",
        "tax_agent": "tax_agent",
        "general_agent": "general_agent"
    }
)

# Connect agent termination paths
workflow.add_edge("journal_agent", END)
workflow.add_edge("tax_agent", END)
workflow.add_edge("general_agent", END)

app = workflow.compile()

# =====================================================================
# 6. RUN INTERACTIVE TERMINAL LOOP
# =====================================================================
async def main():
    print("\n" + "="*60)
    print("💼 Financial Multi-Agent RAG System Online (Llama 3.2 Local)")
    print("Ask about: Journal entries, tax validation, invoices, or operations.")
    print("Type 'exit' to quit.")
    print("="*60 + "\n")
    
    while True:
        try:
            user_question = input("🧠 Question: ")
            if user_question.strip().lower() in ['exit', 'quit']:
                print("\nShutting down pipeline. Goodbye!\n")
                break
            if not user_question.strip():
                continue
                
            initial_state = {"question": user_question}
            final_output = {}
            
            async for output in app.astream(initial_state):
                for node_name, state_update in output.items():
                    final_output.update(state_update)
            
            print(f"\n🤖 System Answer:\n{final_output.get('final_answer')}")
            print("-" * 60 + "\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!\n")
            break

if __name__ == "__main__":
    asyncio.run(main())
