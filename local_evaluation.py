import bz2
import json
import os
from datetime import datetime

from loguru import logger
from openai import APIConnectionError, OpenAI, RateLimitError
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("tokenizer")


def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + IN_CONTEXT_EXAMPLES


def attempt_api_call(client, model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(
                f"API call failed on attempt {attempt + 1}, retrying..."
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None


def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)


def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if "accuracy" in model_resp and (
            (model_resp["accuracy"] is True)
            or (
                isinstance(model_resp["accuracy"], str)
                and model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = 1
        else:
            raise ValueError(
                f"Could not parse answer from response: {model_resp}"
            )

        return answer
    except:
        return -1

def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens"""
    max_token_length = 75
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length+1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction

def generate_predictions(dataset_path, participant_model): 
    
    
    import pickle
    with open('../meta-comphrehensive-rag-benchmark-starter-kit-master/crag_task_1_v1_data', 'rb') as f:
        datas=pickle.load(f)

    predictions = []
    with bz2.open(DATASET_PATH, "rt") as bz2_file:
        for data in tqdm(datas, desc="Generating Predictions"):
            
            
            query = data["query"]
            web_search_results = data["search_results"]
            
            prediction = participant_model.generate_answer(
                query, web_search_results
            )
            
            # trim prediction to 75 tokens
            prediction = trim_predictions_to_max_token_length(prediction)
            predictions.append(
                {
                    "query": query,
                    "ground_truth": str(data["answer"]).strip().lower(),
                    "prediction": str(prediction).strip().lower(),
                }
            )            

    return predictions


def evaluate_predictions(predictions, evaluation_model_name, openai_client):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    system_message = get_system_message()

    for prediction_dict in tqdm(predictions, total=len(predictions), desc="Evaluating Predictions"):
        query, ground_truth, prediction = (
            prediction_dict["query"],
            prediction_dict["ground_truth"],
            prediction_dict["prediction"],
        )

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
            },
        ]
        if prediction == "i don't know" or prediction == "i don't know.":
            n_miss += 1
            continue
        if prediction == ground_truth:
            n_correct_exact += 1
            n_correct += 1
            continue

        response = attempt_api_call(
            openai_client, evaluation_model_name, messages
        )
        if response:
            log_response(messages, response)
            eval_res = parse_response(response)
            if eval_res == 1:
                n_correct += 1

    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_correct_exact,
        "total": n,
    }
    logger.info(results)
    return results


if __name__ == "__main__":
    from models.user_config import UserModel

    DATASET_PATH = "example_data/dev_data.jsonl.bz2"
    EVALUATION_MODEL_NAME = os.getenv(
        "EVALUATION_MODEL_NAME", "gpt-4-0125-preview"
    )

    # Generate predictions
    participant_model = UserModel()
    predictions = generate_predictions(DATASET_PATH, participant_model)

    # Evaluate Predictions
    openai_client = OpenAI()
    evaluation_results = evaluate_predictions(
        predictions, EVALUATION_MODEL_NAME, openai_client
    )
