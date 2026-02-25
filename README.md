# rag-pipeline-comparison

A systematic evaluation framework for comparing Retrieval-Augmented Generation (RAG) pipeline configurations on a SaaS help center knowledge base.

## Overview

This project evaluates how different RAG pipeline configurations impact retrieval and generation quality. Specifically, it tests variations in:

- **Chunk Size**: Different document segmentation sizes (e.g., 256, 512, 1024 tokens)
- **Chunk Overlap**: Overlapping content between chunks to preserve context
- **Retrieval Methods**: Different vector search and ranking strategies

The evaluation uses a custom RAGAS-equivalent framework to assess metrics like:
- Relevance of retrieved documents
- Answer faithfulness to source material
- Context precision and recall
- Overall answer quality

## Project Structure
