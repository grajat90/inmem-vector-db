import numpy as np
import pandas as pd
from models.chunk import Chunk, ChunkMetadata
from models.document import Document, DocumentMetadata
from models.library import Library, LibraryMetadata
import cohere

co = cohere.Client("A1Fi5KBBNoekwBPIa833CBScs6Z2mHEtOXxr52KO")


documents: dict[str, Document] = {}
chunks: list[Chunk] = []

chunks_df = pd.read_csv("chunks_test.csv")
chunk_texts = chunks_df["chunk"].tolist()
document_names = chunks_df["document_name"].tolist()
embeddings_result = co.embed(
    texts=chunk_texts,
    input_type="search_query",
    model="embed-v4.0",
    embedding_types=["float"]
)
embeddings = embeddings_result.embeddings.float

for index, chunk_text in enumerate(chunk_texts):
    embedding = np.array(embeddings[index], dtype=np.float32)
    document_name = document_names[index]
    document = documents.get(document_name)
    if document is None:
        document = Document(
            title=document_name,
            description=document_name,
            chunks=[],
            metadata=DocumentMetadata(
                tags=["test"],
                source="test_source"
            )
        )
        documents[document_name] = document

    chunk = Chunk(
        text=chunk_text,
        embedding=embedding,
        document_id=document.id,
        metadata=ChunkMetadata(
            tags=["test"],
            source="test_source"
        )
    )
    chunks.append(chunk)
    document.chunks.append(chunk.id)

library = Library(
    name="test_library",
    documents={
        document.id: document
        for document in documents.values()
    },
    chunks={
        chunk.id: chunk
        for chunk in chunks
    },
    metadata=LibraryMetadata(
        tags=["test"]
    )
)

query_str = "How to get started with bitcoin mining?"
query_embedding = np.array(co.embed(
    texts=[query_str],
    input_type="search_query",
    model="embed-v4.0",
    embedding_types=["float"]
).embeddings.float[0], dtype=np.float32)
chunks = library.search(query_embedding, k=5)
final_chunks = [chunk.text for chunk in chunks]
print("Search Results:")
for i, chunk_text in enumerate(final_chunks, 1):
    print(f"\n--- Result {i} ---")
    print(chunk_text)
print("\n")
