# ER-RAG: Enhance RAG with ER-Based Unified Modeling of Heterogeneous Data Sources

## Overview

This version of the repository has been tailored for KDD CUP. While this repository provides insights into some components of our approach, the complete ER-RAG codebase is not included here at this time. We are actively exploring ways to share more details about ER-RAG in the future and appreciate your understanding.

## Repository Structure

The base framework for our solution is derived from the [Meta Comprehensive RAG Benchmark Starter Kit](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/tree/master). We have extended and customized this framework to develop our winning solution, particularly focusing on the `models` folder, which contains the core components of our approach.

Here is an overview of the repository structure:

```plaintext
├── models/
│   ├── dummy_model.py        # Overall pipeline implementation
│   ├── Parse.py              # Parse implementation
│   ├── prompt_api.py         # Prompt implementation
│   ├── Retriever.py          # Retriever implementation
│   ├── training_reproduction/ # Fine-tuning code and data
│   ├── pretrain_models/      # Fine-tuned models storage
│   └── processed_data/       # Public data storage
├── ...                       # Other files and folders from the starter kit                    # Other files and folders from the starter kit
```

To use our solution, you need to download the following models and place them in the specified directories:

1. **Meta Llama 3 - 8B Instruct**:
   Download from [Meta Llama 3 - 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and place it in `models/Llama-3-8B-instruct`.

2. **All-MiniLM-L6-v2**:
   Download from [All-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and place it in `models/all-Mini-L6-v2`.

3. **BGE Reranker v2 m3**:
   Download from [BGE Reranker v2 m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) and place it in `models/bge-reranker-v2-m3`.
