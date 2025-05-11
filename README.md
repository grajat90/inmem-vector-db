# In Memory Vector Database

A vector database implementation with embedding-based search capabilities built using FastAPI.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Project Structure](#project-structure)
- **[Design Decisions](#design-decisions)**
    - [API Design](#api-design)
    - [Indexing Strategies](#indexing-strategies)
    - [Implementation Notes](#implementation-notes)

## Overview

This project implements a vector database for efficient similarity search with support for multiple indexing strategies. It features strongly-typed data models, robust validation, and asynchronous persistence.

#### API Docs
You can refer to API docs by going over to `localhost:8000/docs`

## Features

- Document and chunk management with embeddings
- Multiple vector indexing algorithms
- Metadata filtering
- Save/load functionality with persistence to disk
- Asynchronous background operations
- REST API with FastAPI
- Cohere integration for embeddings

## Setup and Installation

### Prerequisites
- Python 3.10+
- Cohere API key

### Local Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd vector-db

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file to add your Cohere API key and other configurations

# Run the application
python -m app.main
```

### Using Environment Variables

The application uses the following environment variables:

```
API_HOST=0.0.0.0
API_PORT=8000
COHERE_API_KEY=your_cohere_api_key
EMBEDDING_MODEL=embed-v4.0
DATA_DIRECTORY=./data
```

## Project Structure

```
.
├── app/                    # Main application package
│   ├── api/                # API functionality
│   │   ├── endpoints/      # API route handlers
│   │   │   ├── libraries.py  # Library management endpoints
│   │   │   ├── documents.py  # Document management endpoints
│   │   │   ├── chunks.py     # Chunk management endpoints
│   │   │   └── search.py     # Search functionality endpoints
│   │   ├── schemas/        # Pydantic models for request/response
│   │   ├── services/       # Business logic services
│   │   ├── exceptions/     # Custom exception handlers
│   │   ├── dependencies.py # FastAPI dependencies
│   │   └── router.py       # Main API router
│   ├── core/               # Core functionality
│   │   ├── models/         # Data models
│   │   │   ├── library.py  # Library model
│   │   │   ├── document.py # Document model
│   │   │   └── chunk.py    # Chunk model
│   │   ├── indexers/       # Vector indexing implementations
│   │   │   ├── flat_index.py # Brute force indexer
│   │   │   ├── hnsw.py     # Hierarchical navigable small world
│   │   │   ├── lsh.py      # Locality Sensitive Hashing
│   │   │   └── indexer.py  # Base indexer interface
│   │   └── embedding.py    # Embedding creation and management
│   ├── config/             # Configuration settings
│   └── main.py             # Application entry point
├── tests/                  # Test suite
├── deployment/             # Deployment configurations
├── .env.example            # Example environment variables
├── .env                    # Environment variables (not in version control)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

# Design Decisions

## API Design
- RESTful API with FastAPI
- CRUD operations for libraries, documents, and chunks
- Vector similarity search endpoints
- Async processing for background tasks
- Comprehensive Swagger documentation at `/docs`


### Data Model Structure

- **Library → Document, Chunk hierarchy**
  - Documents contain chunk IDs only, not full chunks
  - Library contains all chunks, documents, and an indexing scheme
  - Makes it quicker to retrieve any chunk from a library using a simple dict
  - Simpler overall structure

### Technical Requirements

- Static typing throughout (strongly typed)
- FastAPI best practices
- Comprehensive test coverage
- Pydantic validation for request/response models

## Indexing Strategies

The implementation supports multiple indexing approaches:

1. **Brute Force (Flat Index)** - O(n) query time, O(n) space; exact but slow for large datasets
2. **LSH (Locality Sensitivity Hashing)** - O(1) query time, O(n) space; approximate hashing-based search with tunable accuracy
3. **HNSW** - O(log n) query time, O(n log n) space; hierarchical navigable small world with high accuracy and performance

Other indexers considered but not implemented:

- **Annoy** - tree-based approximate search
- **IVF Flat** - forms clusters then performs flat search within clusters


Distance metrics supported:
- Euclidean (L2)
- Cosine similarity
- Manhattan distance
- Dot product

## Implementation Notes

- Race condition avoidance through standard concurrency control algorithms
- Async persistence to disk running in the background
- Efficient metadata filtering capabilities
- Pydantic validation for robust data handling
- Save/load mechanisms for all index types with automatic recovery on restart
- Integration with Cohere for generating embeddings

## References

1. [Weaviate Vector Index Concepts](https://weaviate.io/developers/weaviate/concepts/vector-index)
2. [Vector Indexing: A Roadmap for Vector Databases](https://medium.com/kx-systems/vector-indexing-a-roadmap-for-vector-databases-65866f07daf5)
3. [Understanding Vector DBs Indexing Algorithms](https://gagan-mehta.medium.com/understanding-vector-dbs-indexing-algorithms-ce187dca69c2)