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


---

# Benchmarking Understanding Recall@K

## Overview
Recall@K is a metric commonly used in information retrieval and recommendation systems to evaluate how well a system retrieves relevant items from a larger set. The "@K" indicates that we're only looking at the top K items in the ranked results.

<img src="https://weaviate.io/assets/images/recall-703696c47da2508ebeebb0901ad7addf.jpg">

## Definition
Recall@K measures the proportion of relevant items that are successfully retrieved in the top K results, out of all relevant items that should have been retrieved.

The formula for Recall@K is:

$$ Recall@K = \frac{|\{\text{relevant items}\} \cap \{\text{retrieved items@K}\}|}{|\{\text{relevant items}\}|} $$

Where:
- $|\{\text{relevant items}\} \cap \{\text{retrieved items@K}\}|$ represents the number of relevant items in the top K results
- $|\{\text{relevant items}\}|$ represents the total number of relevant items

## Example: Recall@3
Let's walk through a practical example to understand Recall@3.
More at: https://www.pinecone.io/learn/offline-evaluation/
### Scenario
Imagine a music recommendation system with the following setup:
- User's actual favorite songs (relevant items): "Song A", "Song B", "Song C", "Song D"
- System's top 3 recommendations: "Song A", "Song B", "Song X"

### Calculation
Given:
- $|\{\text{relevant items}\} \cap \{\text{retrieved items@3}\}| = 2$ (Song A, Song B)
- $|\{\text{relevant items}\}| = 4$ (Song A, Song B, Song C, Song D)

$$ Recall@3 = \frac{2}{4} = 0.5 \text{ or } 50\% $$

This means that our system successfully retrieved 50% of the user's favorite songs within its top 3 recommendations.

## Why Use Recall@K?

1. **Limited Display Space**: In real applications, we can only show a limited number of recommendations to users (e.g., first page of search results).

2. **User Attention**: Users typically only look at the first few results, making it crucial to get relevant items in the top positions.

3. **Performance Evaluation**: Helps measure how well a system retrieves relevant items when there's a specific cut-off point.

4. **Trade-off Analysis**: Can be used alongside other metrics (like Precision@K) to understand the balance between finding all relevant items and maintaining result quality.

## Common Values for K
- K = 1: When only the top result matters
- K = 3, 5: For mobile applications or limited screen space
- K = 10: Common for search engine results (first page)
- K = 20, 50: For broader recommendation lists

## Limitations
- Doesn't consider the ordering within the top K results
- Doesn't penalize for irrelevant results
- May not be suitable when the number of relevant items varies greatly between queries

## Example Use Cases
- Search engines
- Product recommendation systems
- Content discovery platforms
- Image retrieval systems
- Document ranking systems