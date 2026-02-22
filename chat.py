from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

VECTORSTORE_PATH = "./vectorstore"
console = Console()

# Load vectorstore
console.print("[bold green]🚀 Loading RAG Agent...[/bold green]")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embeddings
)

# Setup retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Setup LLM
llm = ChatOllama(model="deepseek-r1:14b", temperature=0)

# Custom prompt
prompt = PromptTemplate.from_template("""You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build LCEL RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

console.print("[bold cyan]✅ Agent ready! Type 'exit' to quit.[/bold cyan]\n")

# CLI loop
while True:
    query = console.input("[bold yellow]>:[/bold yellow] ").strip()

    if query.lower() in ["exit", "quit"]:
        console.print("[bold red]Goodbye![/bold red]")
        break

    if not query:
        continue

    with console.status("[bold green]Thinking...[/bold green]"):
        # Get answer
        answer = rag_chain.invoke(query)

        # Get source docs separately
        source_docs = retriever.invoke(query)

    # Print answer
    console.print(Panel(Markdown(answer), title="[bold green]Agent[/bold green]", border_style="green"))

    # Print sources
    if source_docs:
        console.print("[dim]📄 Sources:[/dim]")
        seen = set()
        for doc in source_docs:
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            label = f"{src} (page {page+1})" if page != "" else src
            if label not in seen:
                console.print(f"  [dim]- {label}[/dim]")
                seen.add(label)
    console.print()
