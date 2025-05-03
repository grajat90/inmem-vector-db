import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from models.library import Library, LibraryMetadata
from models.document import Document, DocumentMetadata
from models.chunk import Chunk, ChunkMetadata
from indexers.flat_index import FlatIndexer
from indexers.hsnw import HSNWIndexer
from indexers.lsh import LSHIndexer
from indexers.indexer import DistanceMetric


# Load test data
@pytest.fixture(scope="module")
def test_chunks_df():
    csv_path = os.path.join(os.path.dirname(__file__), "./chunks_test.csv")
    return pd.read_csv(csv_path)


# Create mock embeddings
@pytest.fixture(scope="module")
def mock_embeddings(test_chunks_df):
    embeddings = {}
    for _, row in test_chunks_df.iterrows():
        # Create a stable but unique embedding for each chunk
        # In a real scenario, these would be semantic embeddings from a model
        text_hash = hash(row["chunk"]) % 10000
        embedding = np.array([float(text_hash / 10000)] * 32, dtype=np.float32)  # 32-dim mock embedding
        embeddings[row["chunk"]] = embedding
    return embeddings


# Helper function to create test libraries
@pytest.fixture
def create_test_library(test_chunks_df, mock_embeddings):
    def _create_library(name, indexer_type="flat"):
        # Create a library with the specified indexer
        if indexer_type == "hsnw":
            indexer = HSNWIndexer(name="hsnw_test_indexer")
        elif indexer_type == "lsh":
            indexer = LSHIndexer(name="lsh_test_indexer")
        else:
            indexer = FlatIndexer(name="flat_test_indexer")
        
        # Create library instance
        library = Library(
            name=name,
            documents={},
            chunks={},
            metadata=LibraryMetadata(tags=["test", indexer_type]),
            indexer=indexer
        )
        
        # Group chunks by document name
        document_groups = test_chunks_df.groupby("document_name")
        
        # Add documents and chunks to the library
        for document_name, group in document_groups:
            # Create document
            document = Document(
                title=document_name,
                description=f"Test document for {document_name}",
                chunks=[],
                metadata=DocumentMetadata(
                    source="test",
                    tags=["test"]
                )
            )
            
            # Store document in library
            library.documents[document.id] = document
            
            # Create and add chunks for this document
            for _, row in group.iterrows():
                chunk = Chunk(
                    document_id=document.id,
                    text=row["chunk"],
                    embedding=mock_embeddings[row["chunk"]],
                    metadata=ChunkMetadata(
                        source="test",
                        tags=["test"]
                    )
                )
                
                # Add chunk to library
                library.add_chunk(chunk, document)
        
        return library
    
    return _create_library


# Create a temporary directory for tests
@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


def test_save_load_flat_indexer(create_test_library, temp_dir):
    """Test saving and loading a library with a flat indexer"""
    # Create a test library with a flat indexer
    original_library = create_test_library("Test_Flat_Save_Load", "flat")
    
    # Define a save path
    save_path = os.path.join(temp_dir, "test_flat_library.pkl")
    
    # Save the library
    original_library.save(save_path)
    
    # Verify the file exists
    assert os.path.exists(save_path)
    
    # Create a new library instance and load the saved data
    loaded_library = Library(
        name="Empty_Library",
        documents={},
        chunks={},
        metadata=LibraryMetadata(tags=[]),
        indexer=FlatIndexer()
    )
    loaded_library.load(save_path)
    
    # Verify library metadata
    assert loaded_library.id == original_library.id
    assert loaded_library.name == original_library.name
    assert len(loaded_library.documents) == len(original_library.documents)
    assert len(loaded_library.chunks) == len(original_library.chunks)
    
    # Verify the indexer was loaded correctly
    assert type(loaded_library.indexer) == type(original_library.indexer)
    assert len(loaded_library.indexer.embeddings) == len(original_library.indexer.embeddings)
    
    # Test search functionality on the loaded library
    # Get a random chunk from the library to use as a query
    random_chunk_id = list(loaded_library.chunks.keys())[0]
    query_embedding = loaded_library.chunks[random_chunk_id].embedding
    
    # Search the library
    search_results = loaded_library.search(query_embedding, k=5)
    
    # Verify we get results
    assert len(search_results) > 0
    
    # First result should be the query chunk itself (exact match) for flat indexer
    assert search_results[0].id == random_chunk_id


def test_save_load_hsnw_indexer(create_test_library, temp_dir):
    """Test saving and loading a library with an HSNW indexer"""
    # Create a test library with an HSNW indexer
    original_library = create_test_library("Test_HSNW_Save_Load", "hsnw")
    
    # Define a save path
    save_path = os.path.join(temp_dir, "test_hsnw_library.pkl")
    
    # Save the library
    original_library.save(save_path)
    
    # Verify the file exists
    assert os.path.exists(save_path)
    
    # Create a new library instance and load the saved data
    loaded_library = Library(
        name="Empty_Library",
        documents={},
        chunks={},
        metadata=LibraryMetadata(tags=[]),
        indexer=HSNWIndexer()
    )
    loaded_library.load(save_path)
    
    # Verify library metadata
    assert loaded_library.id == original_library.id
    assert loaded_library.name == original_library.name
    assert len(loaded_library.documents) == len(original_library.documents)
    assert len(loaded_library.chunks) == len(original_library.chunks)
    
    # Verify the indexer was loaded correctly
    assert type(loaded_library.indexer) == type(original_library.indexer)
    assert len(loaded_library.indexer.embeddings) == len(original_library.indexer.embeddings)
    
    # Verify graph structure properties are maintained
    assert loaded_library.indexer._m == original_library.indexer._m
    assert loaded_library.indexer._max_level == original_library.indexer._max_level
    assert loaded_library.indexer._entry_point == original_library.indexer._entry_point
    
    # Test that search returns results (we don't test exact matches due to HNSW's approximate nature)
    random_chunk_id = list(loaded_library.chunks.keys())[0]
    query_embedding = loaded_library.chunks[random_chunk_id].embedding
    
    # Search the library
    search_results = loaded_library.search(query_embedding, k=5)
    
    # Verify we get results
    assert len(search_results) > 0
    # Make sure all returned chunks are valid
    for chunk in search_results:
        assert chunk.id in loaded_library.chunks


def test_save_load_lsh_indexer(create_test_library, temp_dir):
    """Test saving and loading a library with an LSH indexer"""
    # Create a test library with an LSH indexer
    original_library = create_test_library("Test_LSH_Save_Load", "lsh")
    
    # Define a save path
    save_path = os.path.join(temp_dir, "test_lsh_library.pkl")
    
    # Save the library
    original_library.save(save_path)
    
    # Verify the file exists
    assert os.path.exists(save_path)
    
    # Create a new library instance and load the saved data
    loaded_library = Library(
        name="Empty_Library",
        documents={},
        chunks={},
        metadata=LibraryMetadata(tags=[]),
        indexer=LSHIndexer()
    )
    loaded_library.load(save_path)
    
    # Verify library metadata
    assert loaded_library.id == original_library.id
    assert loaded_library.name == original_library.name
    assert len(loaded_library.documents) == len(original_library.documents)
    assert len(loaded_library.chunks) == len(original_library.chunks)
    
    # Verify the indexer was loaded correctly
    assert type(loaded_library.indexer) == type(original_library.indexer)
    assert len(loaded_library.indexer.embeddings) == len(original_library.indexer.embeddings)
    
    # Verify LSH-specific properties are maintained
    assert loaded_library.indexer._hash_size == original_library.indexer._hash_size
    assert loaded_library.indexer._num_tables == original_library.indexer._num_tables
    assert loaded_library.indexer._seed == original_library.indexer._seed
    assert len(loaded_library.indexer._hash_tables) == len(original_library.indexer._hash_tables)
    
    # Test that search returns results
    random_chunk_id = list(loaded_library.chunks.keys())[0]
    query_embedding = loaded_library.chunks[random_chunk_id].embedding
    
    # Search the library
    search_results = loaded_library.search(query_embedding, k=5)
    
    # Verify we get results
    assert len(search_results) > 0
    # Make sure all returned chunks are valid
    for chunk in search_results:
        assert chunk.id in loaded_library.chunks


def test_cross_library_search(create_test_library):
    """Test searching across libraries with different indexers"""
    # Create a library with a flat indexer
    flat_library = create_test_library("Test_Flat_Cross_Search", "flat")
    
    # Create a library with an HSNW indexer
    hsnw_library = create_test_library("Test_HSNW_Cross_Search", "hsnw")
    
    # Create a library with an LSH indexer
    lsh_library = create_test_library("Test_LSH_Cross_Search", "lsh")
    
    # Get a random chunk from the flat library to use as a query
    flat_chunk_id = list(flat_library.chunks.keys())[0]
    query_embedding = flat_library.chunks[flat_chunk_id].embedding
    
    # Search all libraries with the same query
    flat_results = flat_library.search(query_embedding, k=5)
    hsnw_results = hsnw_library.search(query_embedding, k=5)
    lsh_results = lsh_library.search(query_embedding, k=5)
    
    # Verify we get results from all libraries
    assert len(flat_results) > 0
    assert len(hsnw_results) > 0
    assert len(lsh_results) > 0
    
    # Test with different distance metrics
    metrics = [DistanceMetric.EUCLIDEAN, DistanceMetric.COSINE, 
              DistanceMetric.DOT_PRODUCT, DistanceMetric.MANHATTAN]
    
    for metric in metrics:
        # Search in each library with different metrics
        for library, name in [(flat_library, "flat"), 
                              (hsnw_library, "hsnw"),
                              (lsh_library, "lsh")]:
            results = library.indexer.search(query_embedding, k=3, distance_metric=metric)
            
            # Verify we get results with each metric
            assert len(results) > 0, f"No results for {name} with {metric} metric"


def test_save_file_exists(create_test_library, temp_dir):
    """Test that saving a library creates a file with the expected content"""
    # Create a test library
    library = create_test_library("Test_File_Exists", "flat")
    
    # Define a save path
    save_path = os.path.join(temp_dir, "test_file_exists.pkl")
    
    # Save the library
    library.save(save_path)
    
    # Verify the file exists and has content
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0


def test_save_load_with_custom_path(create_test_library, temp_dir):
    """Test saving and loading a library with a custom path"""
    # Create nested directories
    nested_dir = os.path.join(temp_dir, "nested", "directories")
    os.makedirs(nested_dir, exist_ok=True)
    
    # Create a test library
    library = create_test_library("Test_Custom_Path", "flat")
    
    # Define a save path in the nested directory
    save_path = os.path.join(nested_dir, "test_custom_path.pkl")
    
    # Save the library
    library.save(save_path)
    
    # Verify the file exists
    assert os.path.exists(save_path)
    
    # Create a new library instance and load the saved data
    loaded_library = Library(
        name="Empty_Library",
        documents={},
        chunks={},
        metadata=LibraryMetadata(tags=[]),
        indexer=FlatIndexer()
    )
    loaded_library.load(save_path)
    
    # Verify library metadata
    assert loaded_library.id == library.id
    assert loaded_library.name == library.name


def test_corrupt_file_handling(temp_dir):
    """Test handling of corrupt files when loading"""
    # Create a corrupt file
    corrupt_path = os.path.join(temp_dir, "corrupt.pkl")
    with open(corrupt_path, 'wb') as f:
        f.write(b'This is not a valid pickle file')
    
    # Create a new library instance
    library = Library(
        name="Test_Corrupt",
        documents={},
        chunks={},
        metadata=LibraryMetadata(tags=[]),
        indexer=FlatIndexer()
    )
    
    # Attempt to load the corrupt file and expect an exception
    with pytest.raises(Exception):
        library.load(corrupt_path)


# Parameterized test to check all indexers in a single test
@pytest.mark.parametrize("indexer_type", ["flat", "hsnw", "lsh"])
def test_indexer_persistence(create_test_library, temp_dir, indexer_type):
    """Test that all types of indexers persist correctly"""
    # Create a test library with the specified indexer
    library = create_test_library(f"Test_{indexer_type.upper()}_Persistence", indexer_type)
    
    # Define a save path
    save_path = os.path.join(temp_dir, f"test_{indexer_type}_persistence.pkl")
    
    # Save the library
    library.save(save_path)
    
    # Verify the file exists
    assert os.path.exists(save_path)
    
    # Create a new library instance with the appropriate indexer type
    if indexer_type == "hsnw":
        indexer_class = HSNWIndexer
    elif indexer_type == "lsh":
        indexer_class = LSHIndexer
    else:
        indexer_class = FlatIndexer
        
    loaded_library = Library(
        name="Empty_Library",
        documents={},
        chunks={},
        metadata=LibraryMetadata(tags=[]),
        indexer=indexer_class()
    )
    
    # Load the library
    loaded_library.load(save_path)
    
    # Verify the loaded library has the correct indexer type
    assert type(loaded_library.indexer) == type(library.indexer)
    assert len(loaded_library.indexer.embeddings) == len(library.indexer.embeddings)


def test_lsh_multi_probe_functionality():
    """Test the multi-probe functionality of LSH indexer"""
    # Create a simple LSH indexer for testing
    indexer = LSHIndexer(hash_size=4, num_tables=2, seed=42)
    
    # Create a simple test vector
    test_vector = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    
    # Initialize hyperplanes manually
    indexer._dim = 4
    indexer._initialize_hyperplanes()
    
    # Get the hash for the test vector
    hash_value = indexer._hash_vector(test_vector, 0)
    
    # Test multi-probe with different probe counts
    probe_hashes_1 = indexer._multi_probe(hash_value, num_probes=1)
    probe_hashes_2 = indexer._multi_probe(hash_value, num_probes=2)
    
    # The original hash should always be in the result
    assert hash_value in probe_hashes_1
    assert hash_value in probe_hashes_2
    
    # For 1 probe, we should get 1 + 4 = 5 hashes (original + flipping each of the 4 bits)
    assert len(probe_hashes_1) == 5
    
    # For 2 probes, we should get more hashes (original + flipping different bits)
    assert len(probe_hashes_2) > len(probe_hashes_1)
    
    # Each hash in the probes should differ from the original by at most the number of probes
    for p_hash in probe_hashes_1:
        if p_hash != hash_value:
            # Count bit differences
            diff_count = sum(1 for a, b in zip(hash_value, p_hash) if a != b)
            assert diff_count <= 1
    
    for p_hash in probe_hashes_2:
        if p_hash != hash_value:
            # Count bit differences
            diff_count = sum(1 for a, b in zip(hash_value, p_hash) if a != b)
            assert diff_count <= 2