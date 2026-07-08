from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# =====================================================
# Configuration
# =====================================================

PDF_PATH = "sample_pdf.pdf"

CHROMA_DB_PATH = "chroma_db"

COLLECTION_NAME = "pdf_collection"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Word-based chunking (Day 5): sizes are in WORDS, not characters.
# Kept small so each chunk stays under the embedding model's 256-token limit
# (200 words tokenized to ~618 tokens and got silently truncated).
CHUNK_SIZE = 80

CHUNK_OVERLAP = 20


# =====================================================
# Load PDF
# =====================================================

def load_pdf(path):

    print("Loading PDF...")

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    print("PDF Loaded Successfully.")
    print(f"Characters: {len(text)}")

    return text


# =====================================================
# Split Text into Chunks (WORD-BASED)
# =====================================================

def split_text(text, chunk_size=200, overlap=40):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunks.append(" ".join(chunk_words))

        start += chunk_size - overlap

    return chunks


# =====================================================
# Load Embedding Model
# =====================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# =====================================================
# Read PDF
# =====================================================

document = load_pdf(PDF_PATH)

chunks = split_text(
    document,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

print(f"\nCreated {len(chunks)} chunks (word-based).")


# =====================================================
# Generate Embeddings
# =====================================================

print("\nGenerating embeddings...")

embeddings = embedding_model.encode(
    chunks,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Embeddings generated.")


# =====================================================
# Create Chroma Database
# =====================================================

print("\nCreating Chroma database...")

client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

try:
    client.delete_collection(COLLECTION_NAME)
    print("Existing collection deleted.")
except:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

print("Collection created.")


# =====================================================
# Store Chunks
# =====================================================

print("\nSaving chunks...")

ids = []
documents = []
embedding_list = []

for i, chunk in enumerate(chunks):

    ids.append(f"chunk_{i}")

    documents.append(chunk)

    embedding_list.append(
        embeddings[i].tolist()
    )

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embedding_list
)

print(f"Stored {len(chunks)} chunks.")


# =====================================================
# Done
# =====================================================

print("\n" + "=" * 60)
print("Vector Database Created Successfully!")
print("=" * 60)
print(f"Database : {CHROMA_DB_PATH}")
print(f"Collection : {COLLECTION_NAME}")
print("=" * 60)
