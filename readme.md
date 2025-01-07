# Semantic Search Benchmarking data creation (synthetic) with Gemini AI and function calling

For news media dataset. You can extend this to other cases.

This repository provides a framework for **Generative synthetic data creation using semantic search benchmarking**, utilizing Google's **Gemini AI** model and **Vertex AI**. The primary objective is to generate synthetic data (queries and relevant documents) for benchmarking semantic search models.

The data generated can be used to evaluate and compare the effectiveness of various semantic search algorithms using retrieval ranking metrics like recall@k, mrecall@k etc. You can benchmark initially with bruteforce search and then use retrieval engines like BM25, DPR ANN, FAISS, vector databases etc. to benchmark against bruteforce search engine.

The following logic does not guarantee the high quality yield of queries and relevant documents - it should be manually inspected by domain experts for validation.

## Overview

The project involves the following key components:

1. **Load Article Data**: Articles are loaded from a CSV file (`articles.csv`) that contains `article_id` and `all_content`.
2. **Generate Synthetic Data**: For each article, a simple query is generated using the Gemini AI model (`gemini-1.5-flash-002`). The goal is to generate queries that can be answered with relevant paragraphs from the article.
3. **Benchmarking**: The generated data is intended to be used in evaluating and comparing the performance of various **semantic search models**.
4. **Output Data**: The synthetic data is saved in a JSON file, which includes the query, relevant paragraphs, and the associated article ID.

## Features

- **Generates synthetic queries** based on the content of articles.
- Uses **Gemini AI** to generate contextually relevant data for semantic search tasks.
- **Benchmarking dataset** for evaluating semantic search models.
- Outputs data in **JSON format** that includes the query, relevant context, and article ID.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- Google Cloud SDK (for interacting with Vertex AI) and apiplatform