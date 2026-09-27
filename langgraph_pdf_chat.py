import asyncio
import os
from typing import TypedDict
import pypdf
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. READ AND CHUNK THE LOCAL PDF FILE
# =====================================================================
pdf_path = "companypolicy.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(
        f"Missing '{pdf_path}' in this directory! Please create it before running."
    )

print("--- Loading and Parsing PDF File Natively ---")

pdf_pages = []
with open(pdf_path, "rb") as f:
    reader = pypdf.PdfReader(f)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pdf_pages.append(Document(page_content=text, metadata={"source": pdf_path, "page": page_num}))

# Split the extracted PDF text into clean semantic fragments
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
docs = text_splitter.split_documents(pdf_pages)

print(f"-> Successfully chunked PDF into {len(docs)} text fragments.")

# Build the In-Memory vector database using the dedicated embedding model
print("--- Initializing Local Vector Store ---")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)

# Expose a retrieval path to grab the single most relevant paragraph (k=1)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# Initialize the main text synthesis model
llm = ChatOllama(model="llama3.2:latest", temperature=0)


# =====================================================================
# 2. DEFINE STATE AND ROUTING NODES
# =====================================================================
class PDFState(TypedDict):
    question: str
    retrieved_context: str
    final_answer: str

def retrieve_pdf_context_node(state: PDFState):
    user_query = state["question"]
    matched_docs = retriever.invoke(user_query)
    context_text = matched_docs[0].page_content if matched_docs else "No reference found."
    return {"retrieved_context": context_text}

def generate_grounded_answer_node(state: PDFState):
    context = state["retrieved_context"]
    question = state["question"]
    
    prompt = (
        f"You are a helpful document assistant. Answer the user's question based strictly on the PDF context provided below.\n"
        f"If the answer is not in the context, respond with 'I cannot find the answer in the provided document.'\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    
    response = llm.invoke(prompt)
    return {"final_answer": response.content.strip()}


# =====================================================================
# 3. CONSTRUCT THE LANGGRAPH PIPELINE
# =====================================================================
workflow = StateGraph(PDFState)

workflow.add_node("pdf_retriever", retrieve_pdf_context_node)
workflow.add_node("answer_generator", generate_grounded_answer_node)

workflow.add_edge(START, "pdf_retriever")
workflow.add_edge("pdf_retriever", "answer_generator")
workflow.add_edge("answer_generator", END)

app = workflow.compile()


# =====================================================================
# 4. RUN TIME INTERACTIVE CHAT LOOP
# =====================================================================
async def main():
    print("\n" + "="*50)
    print("🤖 Local PDF Chatbot initialized using LangGraph & Llama 3.2")
    print("Type your questions below. Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            # Capture manual human query input from command line
            user_question = input("🧠 Ask a question: ")
            
            # Check exit conditions
            if user_question.strip().lower() in ['exit', 'quit']:
                print("\nGoodbye!\n")
                break
                
            if not user_question.strip():
                continue
                
            print("⏳ Scanning document and thinking...")
            
            # Inject question into the graph pipeline execution context
            initial_input = {"question": user_question}
            final_output = {}
            
            async for output in app.astream(initial_input):
                for node_name, state_update in output.items():
                    final_output.update(state_update)

            print(f"🤖 Response: {final_output.get('final_answer')}")
            print("-" * 50 + "\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!\n")
            break

if __name__ == "__main__":
    asyncio.run(main())
