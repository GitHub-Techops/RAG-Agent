from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_PATH = "./docs"
VECTORSTORE_PATH = "./vectorstore"

print("📂 Loading documents...")
loader = DirectoryLoader(
    DOCS_PATH,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

# Also load .txt files if any
txt_loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
documents += txt_loader.load()

print(f"✅ Loaded {len(documents)} document(s)")

print("✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

print("🧠 Creating embeddings & storing in ChromaDB...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTORSTORE_PATH
)

print("✅ Done! Vectorstore saved. You can now run chat.py")
