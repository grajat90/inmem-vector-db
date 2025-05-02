# StackAI Take Home

### Plan of action:
- Create base models for chunks, docs and library
- Use pydantic for data models
- Form classes with body, embedding and metadata
- Find out best practices to create embedding (use cohere) and metadata
- Create indexer classes with standard functions to add to index, remove, update, rebuild, etc (talk about the different indexers)
- Make a not of race conditions, talk about the ways you avoid them (use standard algos for this)
- apply knn over the indexed points
- Implement crud over these and expose them
- Add a persist to disk async logic that does this in the bg
- Add metadata filtering on top
- Create Python SDK (if time allows)
- create docker image
- create helm chart

### Points to note **Must must**
- Static typing (strongly typed)
- Fastapi best practices **MUST**
- Tests
- Pydantic validation

### Design decisions
- Creating modules for data classes of chunk, document and library
    - Using pydantic
    - Plan to use a lot of validation
- api should sit inside api folder
- trying out a couple of indexers [^1] [^2] [^3]
    - brute force a.k.a flat index
    - IVF Flat (form clusters then flat search within cluster)
    - LSH (locality sensitivity hashing)
    - Annoy (tree based)
    - hsnw (most widely used)
    - inverted index
- For data classes structure is:
    1. Library -> Document, Chunk
        - Documents contain chunk ids only, not full chunks
        - Library contains all chunks, documents and an indexing scheme
        - Would make it quicker to retireve any chunk from a library using a simple dict
        - Simpler
- Added choice for distance heuristic to choose from euclidean l2, cosine distance, manhattan or dot product

### API design choice
- Structure into ~~app dir with~~ api folder containing api routes
- other core functionality in core folder
- co located but separate dependecies file
- create async tasks for reindexing etc
- use pydantic for validations
- @todo: handle pydantic errors better

[^1]: https://weaviate.io/developers/weaviate/concepts/vector-index
[^2]: https://medium.com/kx-systems/vector-indexing-a-roadmap-for-vector-databases-65866f07daf5
[^3]: https://gagan-mehta.medium.com/understanding-vector-dbs-indexing-algorithms-ce187dca69c2