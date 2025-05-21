# benchmark_rag.py
# This script performs a basic benchmark of the RAG service
# using a dataset loaded from the Hugging Face Hub via the 'datasets' library.
# It measures retrieval and generation quality.

import json
import requests
import time
import os
import logging
from collections import Counter
import sys
import argparse # Import argparse for command-line arguments

# Import the datasets library
from datasets import load_dataset, Dataset, DatasetDict

# --- Configuration ---
# Default dataset identifier on Hugging Face Hub for benchmarking
BENCHMARK_DATASET_ID = "LLukas22/nq-simplified"

# Default dataset split to use for benchmarking
# For LLukas22/nq-simplified, 'test' is the evaluation split.
# For 'squad', 'validation' is the evaluation split.
BENCHMARK_DATASET_SPLIT = "test"

# Default cache directory (can be overridden by HF_DATASETS_CACHE environment variable or --cache_dir argument)
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")

RAG_SERVICE_URL = "http://localhost:8000/ask" # URL of your running FastAPI RAG service
DEBUG_RETRIEVER_URL = "http://localhost:8000/debug-retriever" # URL for the debug retriever endpoint

# Number of questions to sample from the dataset for a quick test
# Set to None to run on all questions (can take a long time)
NUM_QUESTIONS_TO_SAMPLE = 100
# ------------------------

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ---------------------

# --- Dataset Loading Function (using Hugging Face datasets) ---
def load_benchmark_dataset(dataset_id, split, cache_dir=None):
    """
    Loads a specific split of a dataset from the Hugging Face Hub.

    Args:
        dataset_id (str): The identifier of the dataset on Hugging Face Hub.
        split (str): The specific split to load (e.g., "train", "validation", "test").
        cache_dir (str, optional): The directory to cache datasets. Defaults to Hugging Face default.

    Returns:
        Dataset: The loaded dataset split object, or None if loading fails.
    """
    logging.info(f"Loading dataset '{dataset_id}' split '{split}' from Hugging Face Hub.")
    if cache_dir:
        logging.info(f"Using cache directory: {cache_dir}")
        os.environ["HF_DATASETS_CACHE"] = cache_dir # Set env var for load_dataset

    try:
        # The load_dataset function automatically handles downloading and caching
        dataset = load_dataset(dataset_id, split=split)
        logging.info(f"✅ Successfully loaded dataset '{dataset_id}' split '{split}'.")
        logging.info(f"Number of examples in the loaded split: {len(dataset)}")
        return dataset

    except FileNotFoundError:
        logging.error(f"❌ Dataset '{dataset_id}' not found on Hugging Face Hub.")
        logging.error("Please check the dataset ID.")
        return None
    except ValueError as ve:
        logging.error(f"❌ Error loading dataset '{dataset_id}' split '{split}': {ve}")
        logging.error(f"Check if the split '{split}' exists for this dataset.")
        return None
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred while loading dataset '{dataset_id}': {e}", exc_info=True)
        return None

# --- Querying and Evaluation Functions (mostly unchanged, adapted for dataset object) ---

def query_rag_service(question, service_url):
    """Sends a query to the RAG service and returns the answer."""
    try:
        response = requests.post(service_url, json={"query": question})
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json().get("answer")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying RAG service: {e}", exc_info=True)
        return None

def get_retrieved_documents(question, debug_retriever_url):
    """Uses the debug retriever endpoint to get the documents retrieved for a query."""
    try:
        response = requests.post(debug_retriever_url, json={"query": question})
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json().get("retrieved_documents", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting retrieved documents from debug endpoint: {e}", exc_info=True)
        return []

def evaluate_answer(generated_answer, ground_truth_answers):
    """
    Performs a basic comparison between the generated and ground truth answers.
    This checks if the generated answer is an exact match or contains any of the ground truth answers.
    SQuAD/NQ can have multiple correct answers.
    """
    if generated_answer is None or not ground_truth_answers:
        return "skipped" # Cannot evaluate if generated is missing or no ground truth answers

    generated_answer_lower = generated_answer.strip().lower()

    # Check for exact match with any ground truth answer
    for gt_answer in ground_truth_answers:
        if gt_answer and gt_answer.strip().lower() == generated_answer_lower:
            return "exact_match"

    # Check if generated answer contains any ground truth answer
    for gt_answer in ground_truth_answers:
         if gt_answer and gt_answer.strip().lower() in generated_answer_lower:
              return "contains_ground_truth"

    # Check if any ground truth answer contains the generated answer (if generated is short)
    if len(generated_answer_lower) > 0 and len(generated_answer_lower) < 50: # Heuristic: only check for short generated answers
         for gt_answer in ground_truth_answers:
              if gt_answer and generated_answer_lower in gt_answer.strip().lower():
                   return "ground_truth_contained"


    return "no_match"


def evaluate_retrieval(retrieved_docs, original_context):
    """
    Checks if any of the retrieved documents contain the original context
    from which the ground truth answer was derived.
    This is a basic check for whether the relevant document chunk was retrieved.
    """
    if not retrieved_docs or not original_context:
        return False # Cannot evaluate if either is missing

    original_context_lower = original_context.strip().lower()

    for doc in retrieved_docs:
        if doc and doc.get("page_content"):
            retrieved_content_lower = doc["page_content"].strip().lower()
            # Check if a large part of the original context is a substring of the retrieved content
            # or vice versa. A simple substring check might be too strict or too lenient depending on chunking.
            # A more robust check would compare embeddings of original context chunks vs retrieved chunks.
            # For this basic benchmark, let's check if a significant portion of the original context
            # is present in any single retrieved document's page_content.
            # We'll check if at least 50% of the original context (by character count)
            # is contained within any single retrieved document chunk.

            original_context_len = len(original_context_lower)
            min_match_len = original_context_len * 0.5 # Require at least 50% match

            # Check if original context is a substring of retrieved content (or vice versa)
            if original_context_lower in retrieved_content_lower or retrieved_content_lower in original_context_lower:
                 return True # Simple case: one is a substring of the other

            # More robust check: find the longest common substring or use sequence matcher
            # For simplicity here, we stick to basic substring checks.
            # A more advanced benchmark would compare embeddings or use fuzzy matching.
            # Also check if the ground truth answer string itself is in the retrieved document
            # (This requires passing ground truth answers, which we don't have access to here
            # without modifying the function signature or passing the whole example.
            # Let's stick to checking the original context for now as a proxy for retrieval relevance).


    return False # Relevant context not found in retrieved documents


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting RAG Benchmarking Script ---")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Benchmark a RAG service using a Hugging Face dataset.")
    parser.add_argument(
        "--dataset_id",
        default=BENCHMARK_DATASET_ID,
        help=f"The identifier of the dataset on Hugging Face Hub (default: {BENCHMARK_DATASET_ID})"
    )
    parser.add_argument(
        "--split",
        "-s",
        default=BENCHMARK_DATASET_SPLIT,
        help=f"The dataset split to use for benchmarking (e.g., 'train', 'validation', 'test') (default: {BENCHMARK_DATASET_SPLIT})"
    )
    parser.add_argument(
        "--cache_dir",
        "-c",
        default=os.environ.get("HF_DATASETS_CACHE", DEFAULT_CACHE_DIR), # Use env var if set, otherwise default
        help=f"The directory to cache datasets (default: {os.environ.get('HF_DATASETS_CACHE', DEFAULT_CACHE_DIR)})"
    )
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=NUM_QUESTIONS_TO_SAMPLE,
        help=f"Number of questions to sample for benchmarking (set to -1 or 0 for all) (default: {NUM_QUESTIONS_TO_SAMPLE})"
    )


    args = parser.parse_args()

    # Load the dataset using the Hugging Face datasets library
    # Pass the cache_dir argument to the loading function
    dataset = load_benchmark_dataset(args.dataset_id, args.split, args.cache_dir)

    if dataset is None:
        logging.critical("Failed to load dataset. Exiting benchmarking.")
        sys.exit(1)

    # --- Prepare Questions and Answers from the loaded dataset ---
    # The structure of the dataset object depends on the dataset.
    # For SQuAD and nq-simplified, examples are dictionaries with 'question', 'context', 'answers'.
    # We need to handle potential variations or missing keys.

    qa_list = []
    try:
        # Iterate through the dataset examples
        for example in dataset:
            question = example.get('question')
            # SQuAD/NQ answers are lists of dictionaries with 'text' and 'answer_start'
            answers_info = example.get('answers', {}) # Get the 'answers' dictionary
            ground_truth_answers = answers_info.get('text', []) # Get the list of answer strings

            # The original context from the dataset example
            original_context = example.get('context')

            if question and ground_truth_answers and original_context:
                qa_list.append({
                    "question": question,
                    "ground_truth_answers": ground_truth_answers, # Store as a list
                    "original_context": original_context
                })
            else:
                logging.warning(f"Skipping example due to missing keys: {example.keys()}")


    except Exception as e:
        logging.critical(f"Error processing dataset examples: {e}", exc_info=True)
        logging.critical("Please ensure the dataset has 'question', 'context', and 'answers' keys.")
        sys.exit(1)


    # --- Sample questions if requested ---
    if args.num_samples > 0 and args.num_samples < len(qa_list):
        import random
        random.seed(42) # for reproducibility
        sampled_qa = random.sample(qa_list, args.num_samples)
        logging.info(f"Sampled {len(sampled_qa)} questions for benchmarking.")
        qa_list = sampled_qa
    elif args.num_samples == -1 or args.num_samples >= len(qa_list):
         logging.info(f"Using all {len(qa_list)} questions for benchmarking.")
         pass # Use all questions
    else: # args.num_samples is 0 or negative but not -1
         logging.warning(f"Invalid number of samples specified: {args.num_samples}. Using all questions.")
         logging.info(f"Using all {len(qa_list)} questions for benchmarking.")
         pass # Use all questions


    if not qa_list:
        logging.warning("No valid question-answer pairs extracted from the dataset. Exiting benchmarking.")
        sys.exit(0)

    total_questions = len(qa_list)
    successful_queries = 0
    evaluation_results = Counter()
    retrieval_success_count = 0
    query_times = []

    logging.info(f"\n--- Running benchmark on {total_questions} questions from '{args.dataset_id}' split '{args.split}' ---")

    for i, qa_pair in enumerate(qa_list):
        question = qa_pair["question"]
        ground_truth_answers = qa_pair["ground_truth_answers"] # Now a list
        original_context = qa_pair["original_context"]

        logging.info(f"\nProcessing question {i+1}/{total_questions}: {question}")
        logging.debug(f"Ground Truth Answers: {ground_truth_answers}")
        # logging.debug(f"Original Context (first 200 chars): {original_context[:200]}...") # Avoid logging full context

        # --- Query the RAG service ---
        start_time = time.time()
        generated_answer = query_rag_service(question, RAG_SERVICE_URL)
        end_time = time.time()

        if generated_answer is not None:
            successful_queries += 1
            query_time = end_time - start_time
            query_times.append(query_time)
            logging.info(f"Generated Answer: {generated_answer}")
            logging.info(f"Query Time: {query_time:.4f} seconds")

            # --- Evaluate Generation ---
            evaluation_result = evaluate_answer(generated_answer, ground_truth_answers)
            evaluation_results[evaluation_result] += 1
            logging.info(f"Generation Evaluation: {evaluation_result}")

            # --- Evaluate Retrieval (Optional, requires debug endpoint) ---
            if DEBUG_RETRIEVER_URL:
                 retrieved_docs = get_retrieved_documents(question, DEBUG_RETRIEVER_URL)
                 # Pass original_context for basic retrieval check
                 retrieval_successful = evaluate_retrieval(retrieved_docs, original_context)
                 if retrieval_successful:
                      retrieval_success_count += 1
                      logging.info("Retrieval Evaluation: SUCCESS (Relevant context/answer found in retrieved docs)")
                 else:
                      logging.warning("Retrieval Evaluation: FAILURE (Relevant context/answer NOT found in retrieved docs)")
            else:
                 logging.warning("Skipping retrieval evaluation: DEBUG_RETRIEVER_URL not provided or accessible.")


        else:
            logging.error("Failed to get a response from the RAG service.")
            evaluation_results["query_failed"] += 1


    # --- Report Metrics ---
    logging.info("\n--- Benchmarking Results ---")
    logging.info(f"Dataset: {args.dataset_id}, Split: {args.split}")
    logging.info(f"Total Questions Processed: {total_questions}")
    logging.info(f"Successful Queries: {successful_queries}")
    logging.info(f"Failed Queries: {total_questions - successful_queries}")

    logging.info("\nGeneration Evaluation:")
    for result, count in evaluation_results.items():
        logging.info(f" - {result}: {count} ({count/total_questions:.2%})")

    if query_times:
        avg_query_time = sum(query_times) / len(query_times)
        logging.info(f"\nAverage Successful Query Time: {avg_query_time:.4f} seconds")
    else:
        logging.warning("\nNo successful queries to calculate average time.")

    if DEBUG_RETRIEVER_URL:
         logging.info(f"\nRetrieval Evaluation (Based on finding original context/answer in retrieved docs):")
         logging.info(f"Successful Retrievals: {retrieval_success_count} ({retrieval_success_count/total_questions:.2%})")
         logging.info(f"Failed Retrievals: {total_questions - retrieval_success_count} ({(total_questions - retrieval_success_count)/total_questions:.2%})")
    else:
         logging.warning("\nRetrieval evaluation skipped.")


    logging.info("\n--- Benchmarking Complete ---")
    logging.info("Review the logs above for detailed results.")

