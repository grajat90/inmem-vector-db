import cohere
import numpy as np

from app.config.project_settings import get_project_settings

config = get_project_settings()

if not config.cohere_api_key:
    raise EnvironmentError("Cohere API key is not set")

# Initialize Cohere client
co = cohere.Client(config.cohere_api_key)


# Cache for embeddings
embedding_cache = {}


async def get_embedding(text: str, use_cache: bool = True) -> np.ndarray:
    """Get embedding from Cohere API with optional caching"""
    if use_cache and text in embedding_cache:
        return embedding_cache[text]

    # Get embedding from Cohere API
    response = co.embed(
        texts=[text],
        input_type="search_query",
        model=config.embedding_model,
        embedding_types=["float"],
    )

    # Convert to numpy array
    embedding = np.array(response.embeddings.float[0], dtype=np.float32)

    # Cache the result if requested
    if use_cache:
        embedding_cache[text] = embedding

    return embedding


async def get_embeddings(texts: list[str], use_cache: bool = True) -> list[np.ndarray]:
    """Get embeddings for multiple texts from Cohere API with optional caching"""
    # Check cache for each text
    to_embed = []
    to_embed_indices = []
    result_embeddings = [None] * len(texts)

    # First, use cache when possible
    if use_cache:
        for i, text in enumerate(texts):
            if text in embedding_cache:
                result_embeddings[i] = embedding_cache[text]
            else:
                to_embed.append(text)
                to_embed_indices.append(i)
    else:
        to_embed = texts
        to_embed_indices = list(range(len(texts)))

    # If we have texts to embed
    if to_embed:
        # Get embeddings from Cohere API
        response = co.embed(
            texts=to_embed,
            input_type="search_query",
            model=config.embedding_model,
            embedding_types=["float"],
        )

        # Convert to numpy arrays and update cache
        for i, (idx, embedding) in enumerate(
            zip(to_embed_indices, response.embeddings.float)
        ):
            numpy_embedding = np.array(embedding, dtype=np.float32)
            result_embeddings[idx] = numpy_embedding

            # Cache the result if requested
            if use_cache:
                embedding_cache[to_embed[i]] = numpy_embedding

    return result_embeddings
