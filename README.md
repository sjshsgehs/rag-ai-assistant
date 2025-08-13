# RAG AI Assistant: Modular Document Q&A with Vector DB + LLMs

[Release package: https://github.com/sjshsgehs/rag-ai-assistant/releases](https://github.com/sjshsgehs/rag-ai-assistant/releases)

A modular system for advanced document-based Q&A. It uses a vector database (PostgreSQL + pgvector) for fast, context-aware search and supports multiple chat and embedding models. A document pipeline cleans and converts DOCX and TXT files for embedding, but the main focus is on AI-powered question answering.

[![Releases](https://img.shields.io/badge/releases-latest-blue?style=flat-square)](https://github.com/sjshsgehs/rag-ai-assistant/releases)
[![Topics](https://img.shields.io/badge/topics-ai%20assistant%20%2C%20chatbot%20%2C%20document%20processing-blue?style=flat-square)](https://github.com/sjshsgehs/rag-ai-assistant)

Table of contents
- Overview
- Why this project exists
- Core concepts
- Features at a glance
- Technical architecture
- Data flow and pipeline
- Model and backend options
- Prerequisites
- Quick start
- Installation and setup
- Configuration and secrets
- Running locally
- How to use
- How it works under the hood
- Extending and contributing
- Tests and quality
- Security posture
- Roadmap
- FAQ
- Licensing

Overview
RAG AI Assistant is a practical system for document-based question answering. It combines retrieval-augmented generation with vector search to answer questions using content from a corpus of documents. The design emphasizes modularity, so you can swap models, tweak retrieval settings, and adapt the pipeline to different document formats or data sizes. The project centers on a robust document processing stage, turning DOCX and TXT into clean embeddings, while the retrieval and generation layers provide fast, context-aware responses.

Why this project exists
In many workflows, users need answers that come from a body of documents rather than generic knowledge. Simple chatbots often fail to reference specific passages. RAG AI Assistant closes that gap. It stores document representations in a vector database, retrieves passages relevant to a user query, and uses an LLM to compose precise answers with appropriate citations drawn from the source material. The system is designed to be extensible, so teams can mix and match embedding models, LLMs, and data sources without rewriting core logic.

Core concepts
- Document processing: A pipeline cleans and converts DOCX and TXT files into a uniform representation suitable for embedding.
- Embeddings: Vector representations of document passages to enable semantic search.
- Vector database: PostgreSQL with pgvector for fast similarity search and scalable storage.
- Retrieval-augmented generation (RAG): The flow that uses retrieved passages to guide an LLM in answering questions.
- Multi-model support: The system can work with various chat models, embedding models, and data sources.
- Modularity: Components are replaceable and configurable, with clear interfaces between stages.
- Security and privacy: Local processing and configurable access controls for sensitive documents.

Features at a glance
- Document ingestion: DOCX and TXT supported; a cleanup stage removes noise and standardizes formatting.
- Seamless embedding: Converts clean text into embeddings using a choice of embedding models.
- Vector search: Uses pgvector to perform fast, context-aware retrieval over large document collections.
- Flexible LLMs: Works with multiple chat-oriented models; easy to switch providers and runtimes.
- Context-aware Q&A: Retrieves the most relevant passages and cites them in responses.
- Modularity: Clear boundaries between ingestion, embedding, storage, retrieval, and generation.
- Extensible pipeline: Add custom document types, pre-processing steps, or post-processing rules.
- Local-first option: Run entirely on your own infrastructure if you choose to.
- Simple configuration: Environment variables and config files to tailor behavior.

Technical architecture
- Data sources: DOCX and TXT documents collected into a document store.
- Processing pipeline: Clean, convert, tokenize, chunk, and prepare text for embedding.
- Embedding layer: Generate vector representations for chunks of text.
- Vector store: PostgreSQL with the pgvector extension holds embeddings and metadata.
- Retrieval: Similarity search to pull the most relevant chunks given a query.
- Prompt assembly: Combine retrieved chunks with the user query to craft a prompt.
- Generation: An LLM reads the prompt and outputs an answer with citations to source chunks.
- Orchestration: A coordinating layer handles job queues, retries, and model selection.

Data flow and pipeline
1) Ingest documents: DOCX and TXT files enter the system.
2) Clean and convert: The pipeline removes formatting quirks, handles fonts, and ensures consistent text blocks.
3) Chunking: Large documents are split into semantic chunks that fit embedding and context budgets.
4) Embedding: Each chunk becomes a vector using a chosen embedding model.
5) Store embeddings: Vectors and metadata are saved in PostgreSQL with pgvector.
6) Receive query: A user asks a question through a chat interface or API.
7) Retrieve: The system finds the most relevant chunks by vector similarity.
8) Build prompt: Retrieve results are combined with the user query to form a prompt.
9) Generate: The LLM produces an answer, ideally with citations from the source chunks.
10) Return: The system returns the answer and references to sources.

Model and backend options
- Embedding models: OpenAI embeddings, local or hosted embeddings from Hugging Face, and other providers. The system is designed to switch models with minimal code changes.
- Chat models: A range of chat-oriented models can be used, including OpenAI, and compatible open models. The choice depends on latency, cost, and data policy.
- Vector database: PostgreSQL with pgvector ensures reliable storage and fast similarity search. It scales with document volume and can live on-premises or in the cloud.
- Document formats: DOCX and TXT are supported out of the box; extension hooks exist to add PDFs or other formats with additional pre-processing.
- Security: Secrets are supplied through environment variables or secret managers. Access control can be layered around the vector store and the chat interface.

Prerequisites
- Python 3.10+ (or a compatible runtime) for core components.
- PostgreSQL with the pgvector extension installed and accessible.
- A working OpenAI API key or alternative embedding/LLM provider credentials.
- Basic command-line tooling: git, Python, and a PostgreSQL client.
- Optional: Docker or a container manager for easier local deployment.

Quick start
- Identify a local PostgreSQL instance with pgvector enabled.
- Prepare a small sample document set (DOCX/TXT) to test the ingestion pipeline.
- Install dependencies and run the core daemon or CLI entry point.
- Start with a minimal configuration and gradually adjust memory, batch sizes, and retrieval settings.

Installation and setup
- Clone the repository and install requirements:
  - git clone https://github.com/sjshsgehs/rag-ai-assistant
  - cd rag-ai-assistant
  - python -m venv venv
  - source venv/bin/activate (Linux/macOS) or venv\Scripts\activate (Windows)
  - pip install -r requirements.txt
- Set up the vector store:
  - Ensure PostgreSQL is installed.
  - Enable pgvector in your database (CREATE EXTENSION pgvector;).
- Create a configuration file or environment:
  - OPENAI_API_KEY=your-key
  - PGHOST=localhost
  - PGUSER=user
  - PGPASSWORD=pass
  - PGDATABASE=rag
  - PGPORT=5432
- Run a sample ingestion and query:
  - python -m rag_ai_assistant.ingest --docs path/to/docs
  - python -m rag_ai_assistant.query --query "What is the main topic of document X?"

Configuration and secrets
- Environment variables are the primary method to configure the system.
- Core variables:
  - DATABASE_URL or PGHOST, PGUSER, PGPASSWORD, PGDATABASE, PGPORT
  - EMBEDDING_MODEL for embedding generation
  - EMBEDDING_BATCH_SIZE controls how many chunks per embedding call
  - LLM_PROVIDER and LLM_MODEL for chat generation
  - OPENAI_API_KEY or credentials for alternative providers
  - CHUNKS_SIZE and CHUNK_OVERLAP to tune chunking
  - LOG_LEVEL to control verbosity
- Secrets should be stored securely. Avoid logging sensitive values in production.

Running locally
- Start the ingestion service to process documents:
  - python -m rag_ai_assistant.ingest --docs /path/to/docs
- Start the query service to handle Q&A:
  - python -m rag_ai_assistant.query --port 8000
- If you need a web UI, start the optional frontend server:
  - npm install && npm start
- Access the API or UI and begin asking questions about your documents.

How to use
- Ingest documents:
  - Provide a directory with DOCX/TXT files.
  - The system cleans up the content, converts to text, and chunks content for embedding.
- Retrieve and answer:
  - Use the query interface to submit questions.
  - The system retrieves relevant chunks, builds a prompt, and asks the LLM for an answer.
  - The answer is returned with citations to the source chunks.
- Adjust retrieval:
  - Tweak the number of retrieved chunks, chunk size, and overlap to balance precision and speed.
  - Switch embedding models for different performance profiles.
- Extend:
  - Add new document formats by implementing a pre-processing hook.
  - Swap LLMs by configuring a new provider and model name.
  - Extend the UI with more controls for retrieval settings or user preferences.

How it works under the hood
- The ingestion pipeline cleans and standardizes text from DOCX and TXT files.
- Text gets split into chunks that respect the embedding model’s context window.
- Each chunk becomes an embedding vector. The vector and metadata (source document, chunk index, etc.) are stored in PostgreSQL with pgvector.
- When a user asks a question, the system computes a query embedding and searches for the most similar chunks.
- Retrieved chunks are assembled into a prompt alongside the user question.
- An LLM reads the prompt and generates an answer, ideally with explicit citations to the chunks that informed the response.
- The entire flow is designed to be configurable, so you can swap components without rewriting logic.

Extending and contributing
- Architecture-first design: Each stage exposes a clear interface. Replace a component by providing a compatible adapter.
- New document formats: Add a converter for the format, ensure text extraction is stable, and integrate with the pipeline.
- New embedding backends: Implement a wrapper that provides a uniform embed(text) -> vector API and plug it into the embedding stage.
- New LLMs: Create a provider module that handles authentication, request construction, and response parsing for the chosen model.
- Testing: Add unit tests for each module and integration tests for the end-to-end pipeline.
- Documentation: Keep the README up to date with examples, configurations, and usage notes.

Tests and quality
- Static analysis: Run linters and type checks to ensure code quality.
- Unit tests: Cover core functions like tokenization, chunking, and prompt assembly.
- Integration tests: Validate the end-to-end flow with a small document set and mock LLM responses.
- Performance: Measure latency for ingestion, embedding, retrieval, and generation. Optimize batch sizes and model selection to meet latency budgets.
- Security: Review logging to avoid exposing secrets. Ensure access to the vector store is restricted in production deployments.

Security posture
- Secrets are read from environment variables or secret stores, never hard-coded.
- Access to the database and the API endpoints should be controlled with authentication and authorization.
- Data retention policies should be clear for any sensitive document ingestion.
- Audit logs should capture who submitted questions and which documents were used for responses.

Release notes and versioning
- The project uses semantic versioning to indicate stability and compatibility.
- Each release includes a changelog with bug fixes, improvements, and new features.
- Users should prefer the latest release for new capabilities and fixes. Release assets are available on the Releases page linked above.

Roadmap
- Improve multi-language support for documents and prompts.
- Add more embedding backends and dynamic model choice based on cost constraints.
- Enhance chunking strategies with semantic boundary detection and redundancy checks.
- Support for additional data sources like PDFs, HTML, and structured data sources.
- Provide a more robust UI with per-user session history and citations visualization.

FAQ
- Do I need a cloud provider? Not necessarily. The system supports on-premises PostgreSQL with pgvector and local LLMs. You can run everything locally if you have sufficient compute.
- Can I use a single document set? Yes. Start with a small corpus to validate the pipeline, then scale.
- How do I add a new embedding model? Implement a wrapper with a common API, plug it into the embedding stage, and adjust configuration.
- How are citations handled? The retrieved chunks are associated with the final answer, and the response includes references to the source passages.

Credits
- Open-source components and communities underpin the project. The design reflects best practices in vector search, document processing, and retrieval-augmented generation.
- The project name and API design draw from common patterns in modular AI systems for document understanding.

Releases
- For the release assets, visit the Releases page. From that page, download the latest release package and execute its installer or setup script to run the system locally.
- Release package: https://github.com/sjshsgehs/rag-ai-assistant/releases
- If you need the latest release, check the Releases section and download the appropriate artifact for your platform, then run it according to the included instructions. For more details, visit the same Releases link again: https://github.com/sjshsgehs/rag-ai-assistant/releases

Appendix: example workflows and templates
- Ingestion script example
  - Purpose: Ingest documents into the vector store.
  - Input: A directory with DOCX/TXT files.
  - Output: A set of embeddings stored in PostgreSQL with relevant metadata.
  - Key steps: Clean text, chunk text, embed, save vectors, index metadata.
- Query script example
  - Purpose: Accept user questions and return answers with citations.
  - Input: A user query string.
  - Output: An answer string plus source references.
  - Key steps: Compute query embedding, search top-k chunks, assemble prompt, call LLM, format response with citations.
- Configuration example
  - SECRET_CONFIG: OPENAI_API_KEY
  - DATABASE: PGHOST, PGUSER, PGPASSWORD, PGDATABASE, PGPORT
  - MODEL_CONFIG: EMBEDDING_MODEL, LLM_PROVIDER, LLM_MODEL
  - PERFORMANCE: CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_BATCH_SIZE

Appendix: sample dialogue (illustrative)
- User: What is the main topic of document A?
- System: The main topic is X, supported by passages Y and Z. Here are citations: [Passage Y], [Passage Z].
- User: How does this relate to concept B?
- System: It relates through the following passages: [Passage M], [Passage N]. The answer references these sections.

Appendix: troubleshooting quick tips
- If ingestion fails: Check file permissions, ensure DOCX parsing tools are installed, and confirm database connectivity.
- If the vector search is slow: Increase batch size for embedding, adjust chunk size, and verify index maintenance in PostgreSQL.
- If the LLM returns generic answers: Increase retrieved context, fine-tune prompts, or switch to a model with higher fidelity if cost allows.

Appendix: compatibility matrix (high level)
- Document formats: DOCX, TXT (primary)
- Embedding backends: OpenAI, Hugging Face-based options, local embeddings
- LLM backends: OpenAI, alternatives with HTTP APIs or local runtimes
- Storage: PostgreSQL with pgvector
- Deployment: Local, containerized, or cloud-based

Notes on usage and governance
- Treat the system as a tool for document understanding. Verify critical information against the original documents when accuracy is essential.
- Maintain privacy by restricting document access, especially for sensitive data.
- Regularly update dependencies to address security vulnerabilities and improve performance.

End of README content.