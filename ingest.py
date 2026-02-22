from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

DOCS_PATH = "./docs"
VECTORSTORE_PATH = "./vectorstore"

# ─────────────────────────────────────────
# ADD YOUR URLs HERE
# ─────────────────────────────────────────
WEB_URLS = [
    "https://developer.hashicorp.com/terraform/intro",
    "https://developer.hashicorp.com/terraform/language",
    "https://developer.hashicorp.com/terraform/cli",
    "https://developer.hashicorp.com/terraform/install",
]
# ─────────────────────────────────────────

all_documents = []

# ── Load Local PDFs ──
if os.path.exists(DOCS_PATH) and os.listdir(DOCS_PATH):
    print("📂 Loading local PDF documents...")
    pdf_loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()
    print(f"  ✅ Loaded {len(pdf_docs)} PDF page(s)")
    all_documents.extend(pdf_docs)

    print("📂 Loading local TXT documents...")
    txt_loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
    txt_docs = txt_loader.load()
    print(f"  ✅ Loaded {len(txt_docs)} TXT file(s)")
    all_documents.extend(txt_docs)
else:
    print("⚠️  No local docs found, skipping local loading.")

# ── Load Web URLs ──
if WEB_URLS:
    print("\n🌐 Loading web documentation...")
    for url in WEB_URLS:
        try:
            print(f"  Fetching: {url}")
            web_loader = WebBaseLoader(url)
            web_docs = web_loader.load()
            print(f"  ✅ Loaded {len(web_docs)} page(s) from {url}")
            all_documents.extend(web_docs)
        except Exception as e:
            print(f"  ❌ Failed to load {url}: {e}")

if not all_documents:
    print("\n❌ No documents loaded. Please add local files or valid URLs.")
    exit(1)

print(f"\n📊 Total documents loaded: {len(all_documents)}")

# ── Split into chunks ──
print("\n✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_documents)
print(f"✅ Created {len(chunks)} chunks")

# ── Embed & Store ──
print("\n🧠 Creating embeddings & storing in ChromaDB...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTORSTORE_PATH
)

print("\n✅ Done! Vectorstore updated. You can now run chat.py")