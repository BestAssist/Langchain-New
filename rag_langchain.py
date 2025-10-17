import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain.tools.retriever import create_retriever_tool

# --- 0️⃣ Set your Google API key ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyAn466mU3IsiEDSNcBFTd-PLvssxSWIEpA"

# --- 1️⃣ Setup LLM and embeddings (Gemini) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- 2️⃣ Create sample shopping catalog ---
docs = [
    Document(page_content="Nike Zoom Fly 5, waterproof running shoe, price $120."),
    Document(page_content="Adidas RunFast, lightweight trainer, price $90, not waterproof."),
    Document(page_content="Puma AquaShield, waterproof trail shoe, price $95."),
    Document(page_content="Reebok CloudStep, cushioned comfort shoe, price $85, not waterproof.")
]

# --- 3️⃣ Build the vectorstore ---
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 4️⃣ Wrap retriever as a LangGraph tool ---
retriever_tool = create_retriever_tool(
    retriever,
    name="product_search",
    description="Search product catalog for relevant items based on query."
)

retrieval_node = ToolNode(tools=[retriever_tool])

# --- 5️⃣ Define the LangGraph workflow ---
def summarize_results(state):
    """Generate the final answer using retrieved context."""
    query = state["input"]
    retrieved_docs = state["product_search"]
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"Answer the user query using this context:\n\n{context}\n\nUser query: {query}"
    result = llm.invoke(prompt)
    return {"output": result.content}

graph = StateGraph()
graph.add_node("retrieval", retrieval_node)
graph.add_node("generate", summarize_results)
graph.add_edge("retrieval", "generate")
graph.set_entry_point("retrieval")
graph.set_finish_point("generate")

shopping_assistant = graph.compile()

# --- 6️⃣ Run the graph ---
response = shopping_assistant.invoke({"input": "Which waterproof running shoes cost under $100?"})
print(response["output"])
