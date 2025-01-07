import vertexai
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)
from google.cloud import aiplatform
import pandas as pd
import json

def init_vertex_ai(project: str, location: str):
    aiplatform.init(project=project, location=location)

def load_article_data(file_path: str, sample_size: int = 5, random_state: int = 0):
    df = pd.read_csv(file_path, usecols=['article_id', 'all_content'])
    df.dropna(inplace=True)
    df = df[df['all_content'].str.split().str.len() >= 500]
    subset = df.sample(sample_size, random_state=random_state)
    return subset

def init_model(model_name: str) -> GenerativeModel:
    return GenerativeModel(model_name)

def create_function_declaration():
    return FunctionDeclaration(
        name="generate_synthetic_data",
        description="Generate synthetic data containing queries and relevant documents.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "relevant_doc": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["query", "relevant_doc"],
        }
    )

def create_synthetic_data_tool(function_declaration):
    return Tool(
        function_declarations=[function_declaration],
    )

def create_user_prompt(article_text: str):
    return Content(
        role="user",
        parts=[
            Part.from_text(
                f"Using the provided article, craft a simple query that can be answered effectively with three relevant paragraphs or context through the 'generate_synthetic_data' function.\n\n{article_text}"
            ),
        ],
    )

def generate_synthetic_data(model: GenerativeModel, user_prompt_content: Content, synthetic_data_tool: Tool):
    response = model.generate_content(
        user_prompt_content, generation_config=GenerationConfig(temperature=0), tools=[synthetic_data_tool],
    )
    return response

def extract_response_data(response):
    return {
        "query": response.candidates[0].content.parts[0].function_call.args['query'],
        "relevant_docs": [str(item) for item in response.candidates[0].content.parts[0].function_call.args['relevant_doc']]
    }

def main():
    init_vertex_ai(project="sandbox", location="us-central1")

    articles = load_article_data("articles.csv")
    model = init_model("gemini-1.5-flash-002")

    generate_synthetic_data_func = create_function_declaration()
    synthetic_data_tool = create_synthetic_data_tool(generate_synthetic_data_func)

    data = []
    for _, row in articles.iterrows():
        article_id = row['article_id']
        article_text = row['all_content']
        
        user_prompt_content = create_user_prompt(article_text)
        
        response = generate_synthetic_data(model, user_prompt_content, synthetic_data_tool)
        response_data = extract_response_data(response)
        response_data['article_id'] = article_id  # Add article_id to the response data
        
        data.append(response_data)

    return data



if __name__ == "__main__":
    generated_data = main()
    with open('synthetic_data.json', 'w') as json_file:
        json.dump(generated_data, json_file, indent=4)
    print("Data successfully written to 'synthetic_data.json'")