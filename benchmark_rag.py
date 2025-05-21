# benchmark_rag.py
# This script performs a basic benchmark of the RAG service
# using a dataset loaded from the Hugging Face Hub via the 'datasets' library
# OR from local Q&A JSON files.
# It measures retrieval and generation quality.

import json
import requests
import time
import os
import logging
from collections import Counter
import sys
import argparse
import datetime # For unique output filenames
import random # For shuffling questions

# Import the datasets library (make sure you've run: pip install datasets)
from datasets import load_dataset, Dataset, DatasetDict

# --- Configuration ---
# Default dataset identifier on Hugging Face Hub for benchmarking
BENCHMARK_DATASET_ID = "LLukas22/nq-simplified"

# Default dataset split to use for benchmarking
BENCHMARK_DATASET_SPLIT = "test"

# Default cache directory (can be overridden by HF_DATASETS_CACHE environment variable or --cache_dir argument)
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")

# Default directory for local Q&A JSON files (e.g., where generate_qna.py outputs)
DEFAULT_LOCAL_QNA_DIR = "generated_qna_data" # Assuming this is where your Q&A JSONs are

# URL of your running FastAPI RAG service
RAG_SERVICE_URL = "http://localhost:8000/ask" # As per your script
DEBUG_RETRIEVER_URL = "http://localhost:8000/debug-retriever" # URL for the debug retriever endpoint

# Number of questions to sample from the dataset for a quick test
# Set to -1 to run on all questions
NUM_QUESTIONS_TO_SAMPLE = 100
# ------------------------

# --- Logging Setup ---
# Changed individual item/file skips to DEBUG level to reduce console noise by default
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ---------------------

# --- Data Loading Functions ---

def load_huggingface_dataset(dataset_id: str, split: str, cache_dir: str = None):
    """
    Loads a specific split of a dataset from the Hugging Face Hub.
    """
    logging.info(f"Loading dataset '{dataset_id}' split '{split}' from Hugging Face Hub.")
    if cache_dir:
        logging.info(f"Using cache directory: {cache_dir}")
        os.environ["HF_DATASETS_CACHE"] = cache_dir # Set env var for load_dataset

    try:
        dataset = load_dataset(dataset_id, split=split)
        logging.info(f"✅ Successfully loaded dataset '{dataset_id}' split '{split}'.")
        logging.info(f"Number of examples in the loaded split: {len(dataset)}")
        return dataset

    except Exception as e:
        logging.error(f"❌ An error occurred while loading dataset '{dataset_id}' split '{split}': {e}", exc_info=True)
        logging.error("Please ensure the dataset ID and split are correct, and 'datasets' library is installed.")
        return None

def load_local_qna_from_directory(directory_path: str) -> list[dict]:
    """
    Loads Q&A pairs from all JSON files in a specified directory.
    Assumes each JSON file contains a list of objects with "question", "answer",
    "source_text_snippet", and "id" keys.
    """
    all_qna_pairs = []
    logging.info(f"Loading local Q&A from directory: {directory_path}")
    if not os.path.isdir(directory_path):
        logging.error(f"Directory not found: {directory_path}")
        return []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            # Ensure all expected keys are present
                            if all(k in item for k in ["question", "answer", "source_text_snippet", "id"]):
                                all_qna_pairs.append({
                                    "question": item["question"].strip(),
                                    "ground_truth_answers": [item["answer"].strip()], # Wrap answer in a list for consistency
                                    "original_context": item["source_text_snippet"].strip(), # Use the snippet as context
                                    "document_id": item["id"].strip(), # Use the document ID
                                    "source_file": os.path.basename(file_path) # Track original file
                                })
                            else:
                                missing_keys = [k for k in ["question", "answer", "source_text_snippet", "id"] if k not in item]
                                # Changed to logging.debug: These will only show if logging level is DEBUG
                                logging.debug(f"Skipping malformed item in {filename} (missing keys: {missing_keys}): {item}")
                    else:
                        # Changed to logging.debug: These will only show if logging level is DEBUG
                        logging.debug(f"Skipping malformed file {filename}: Expected a JSON array, got {type(data)}")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from {filename}: {e}")
            except Exception as e:
                logging.error(f"Error reading file {filename}: {e}")
    
    logging.info(f"Loaded {len(all_qna_pairs)} Q&A pairs from local files.")
    return all_qna_pairs

# --- Querying and Evaluation Functions ---

def query_rag_service(question, service_url):
    """Sends a query to the RAG service and returns the answer."""
    try:
        response = requests.post(service_url, json={"query": question}, timeout=60) # Add a timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json().get("answer")
    except requests.exceptions.Timeout:
        logging.error(f"RAG service timed out after 60 seconds for query: '{question[:50]}...'")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying RAG service for '{question[:50]}...': {e}", exc_info=False) # Suppress full traceback for brevity in logs
        return None

def get_retrieved_documents(question, debug_retriever_url):
    """Uses the debug retriever endpoint to get the documents retrieved for a query."""
    if not debug_retriever_url:
        return [] # No debug URL, return empty list

    try:
        response = requests.post(debug_retriever_url, json={"query": question}, timeout=60) # Add a timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json().get("retrieved_documents", [])
    except requests.exceptions.Timeout:
        logging.error(f"Debug retriever timed out after 60 seconds for query: '{question[:50]}...'")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting retrieved documents for '{question[:50]}...': {e}", exc_info=False) # Suppress full traceback
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


def evaluate_retrieval(retrieved_docs: list[dict], original_context: str | None, ground_truth_answers: list[str], document_id: str | None):
    """
    Checks if any of the retrieved documents contain the relevant information.
    Prioritizes matching by document_id, then original_context, then ground_truth_answers.
    
    Args:
        retrieved_docs (list[dict]): List of retrieved documents, each with 'page_content' and 'metadata'.
        original_context (str | None): The source_text_snippet from the Q&A, or None for some HF datasets.
        ground_truth_answers (list[str]): The expected answers.
        document_id (str | None): The ID from the Q&A entry (e.g., "Documentum for Research and Development 16_q_1_0").
    """
    if not retrieved_docs:
        return False # No documents retrieved

    # 1. Check for exact document_id match in retrieved document metadata
    # Assuming 'id' in Q&A matches 'id' or 'source' or 'file_id' in retrieved doc metadata
    if document_id:
        doc_id_lower = document_id.lower()
        for doc in retrieved_docs:
            if doc and "metadata" in doc and isinstance(doc["metadata"], dict):
                # Check common metadata keys for document ID
                if doc_id_lower == doc["metadata"].get("id", "").lower():
                    logging.debug(f"Retrieval SUCCESS: Matched by document_id '{document_id}' via 'id' metadata.")
                    return True
                if doc_id_lower == doc["metadata"].get("source", "").lower(): # Sometimes source contains a unique ID
                    logging.debug(f"Retrieval SUCCESS: Matched by document_id '{document_id}' via 'source' metadata.")
                    return True
                # If your indexer stores the full path or filename in metadata, you might need to adjust
                # e.g., if 'id' is 'Documentum for R&D 16', and metadata.source is 'path/to/Documentum for R&D 16.pdf'
                # you might need to check if doc_id_lower in metadata.get("source", "").lower()
                # For now, let's stick to exact matches on common ID fields.

    # 2. Check against original_context (source_text_snippet) in page_content
    if original_context:
        original_context_lower = original_context.strip().lower()
        # Require at least 50% of the original context (by character count) to be present
        min_match_len = len(original_context_lower) * 0.5 

        for doc in retrieved_docs:
            if doc and doc.get("page_content"):
                retrieved_content_lower = doc["page_content"].strip().lower()
                # Check if original context is a substring of retrieved content (or vice versa)
                if original_context_lower in retrieved_content_lower or retrieved_content_lower in original_context_lower:
                    logging.debug(f"Retrieval SUCCESS: Matched by original_context substring.")
                    return True
                # A more robust check for partial matches could use sequence matching or embeddings.
                # For this benchmark, substring is a reasonable starting point.

    # 3. Check if any ground truth answer is present in retrieved content (as a fallback)
    if ground_truth_answers:
        for gt_answer in ground_truth_answers:
            if gt_answer: # Ensure the ground truth answer is not empty
                gt_answer_lower = gt_answer.strip().lower()
                if not gt_answer_lower: continue 

                for doc in retrieved_docs:
                    if doc and doc.get("page_content"):
                        retrieved_content_lower = doc["page_content"].strip().lower()
                        if gt_answer_lower in retrieved_content_lower:
                            logging.debug(f"Retrieval SUCCESS: Matched by ground_truth_answer substring.")
                            return True
                            
    return False # Relevant information not found in retrieved documents


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting RAG Benchmarking Script ---")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Benchmark a RAG service using local Q&A JSON files or Hugging Face datasets.")
    parser.add_argument(
        "--data_source_type",
        type=str,
        choices=["huggingface", "local_qna"],
        required=True,
        help="Type of Q&A data source: 'huggingface' for a Hugging Face dataset, or 'local_qna' for local JSON files."
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=BENCHMARK_DATASET_ID,
        help=f"Required if --data_source_type is 'huggingface'. The identifier of the dataset on Hugging Face Hub (default: {BENCHMARK_DATASET_ID})"
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default=BENCHMARK_DATASET_SPLIT,
        help=f"Required if --data_source_type is 'huggingface'. The dataset split to use (default: {BENCHMARK_DATASET_SPLIT})"
    )
    parser.add_argument(
        "--cache_dir",
        "-c",
        type=str,
        default=os.environ.get("HF_DATASETS_CACHE", DEFAULT_CACHE_DIR),
        help=f"The directory to cache Hugging Face datasets (default: {os.environ.get('HF_DATASETS_CACHE', DEFAULT_CACHE_DIR)})"
    )
    parser.add_argument(
        "--local_qna_dir",
        type=str,
        default=DEFAULT_LOCAL_QNA_DIR,
        help=f"Required if --data_source_type is 'local_qna'. Path to the directory containing your local Q&A JSON files (default: {DEFAULT_LOCAL_QNA_DIR})."
    )
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=NUM_QUESTIONS_TO_SAMPLE,
        help=f"Number of questions to sample for benchmarking (set to -1 for all) (default: {NUM_QUESTIONS_TO_SAMPLE})"
    )
    parser.add_argument(
        "--rag_service_url",
        type=str,
        default=RAG_SERVICE_URL,
        help=f"URL of your RAG service endpoint (default: {RAG_SERVICE_URL})."
    )
    parser.add_argument(
        "--debug_retriever_url",
        type=str,
        default=DEBUG_RETRIEVER_URL,
        help=f"URL of your RAG service's debug retriever endpoint (default: {DEBUG_RETRIEVER_URL}). Set to empty string '' to skip retrieval evaluation."
    )


    args = parser.parse_args()

    # Update global URLs if provided
    RAG_SERVICE_URL = args.rag_service_url
    DEBUG_RETRIEVER_URL = args.debug_retriever_url


    # --- Load Q&A Data based on source type ---
    qa_list = []
    if args.data_source_type == "huggingface":
        dataset = load_huggingface_dataset(args.dataset_id, args.split, args.cache_dir)
        if dataset is None:
            logging.critical("Failed to load Hugging Face dataset. Exiting benchmarking.")
            sys.exit(1)
        
        # Prepare qa_list from Hugging Face dataset
        try:
            for example in dataset:
                question = example.get('question')
                answers_info = example.get('answers', {}) 
                ground_truth_answers = answers_info.get('text', []) 
                original_context = example.get('context')
                # Hugging Face datasets generally don't have a direct 'document_id' like your local Q&A
                document_id = None 

                if question and ground_truth_answers: # original_context is optional for this benchmark
                    qa_list.append({
                        "question": question,
                        "ground_truth_answers": ground_truth_answers,
                        "original_context": original_context,
                        "document_id": document_id, # Will be None for HF datasets unless manually added
                        "source_file": f"HF: {args.dataset_id}/{args.split}"
                    })
                else:
                    logging.warning(f"Skipping HF example due to missing question or answers: {example.keys()}")
        except Exception as e:
            logging.critical(f"Error processing Hugging Face dataset examples: {e}", exc_info=True)
            sys.exit(1)

    elif args.data_source_type == "local_qna":
        qa_list = load_local_qna_from_directory(args.local_qna_dir)

    if not qa_list:
        logging.warning("No valid question-answer pairs loaded from any source. Exiting benchmarking.")
        sys.exit(0)

    # --- SHUFFLE THE QUESTIONS RANDOMLY ---
    # Shuffle the entire list of questions to ensure random order on each run
    random.shuffle(qa_list) 

    # --- Sample questions if requested ---
    if args.num_samples > 0 and args.num_samples < len(qa_list):
        # Taking the first N questions from the shuffled list
        sampled_qa = qa_list[:args.num_samples] 
        logging.info(f"Sampled {len(sampled_qa)} questions for benchmarking.")
        qa_list = sampled_qa
    elif args.num_samples == -1 or args.num_samples >= len(qa_list):
        logging.info(f"Using all {len(qa_list)} questions for benchmarking.")
        pass
    else: # args.num_samples is 0 or negative but not -1
        logging.warning(f"Invalid number of samples specified: {args.num_samples}. Using all questions.")
        logging.info(f"Using all {len(qa_list)} questions for benchmarking.")
        pass


    total_questions = len(qa_list)
    successful_queries = 0
    evaluation_results = Counter()
    retrieval_success_count = 0
    query_times = []
    
    # Store full results for detailed output (still needed for final pretty-printed JSON)
    full_results_data = []

    logging.info(f"\n--- Running benchmark on {total_questions} questions from '{args.data_source_type}' ---")

    # --- Prepare incremental results file (JSONL) ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    incremental_output_filename = f"benchmark_results_{args.data_source_type}_{timestamp}.jsonl"
    logging.info(f"Incremental results will be saved to: {incremental_output_filename}")

    try:
        # Open the JSONL file in append mode. It will be created if it doesn't exist.
        with open(incremental_output_filename, 'a', encoding='utf-8') as f_incremental:
            for i, qa_pair in enumerate(qa_list):
                question = qa_pair["question"]
                ground_truth_answers = qa_pair["ground_truth_answers"] # Now a list
                original_context = qa_pair.get("original_context") # Can be None for HF datasets
                document_id = qa_pair.get("document_id") # Can be None for HF datasets
                source_info = qa_pair.get("source_file", "N/A") # For local Q&A files

                current_result = {
                    "query_number": i + 1,
                    "question": question,
                    "ground_truth_answers": ground_truth_answers,
                    "original_context_provided": (original_context is not None),
                    "document_id_provided": (document_id is not None),
                    "source_info": source_info,
                    "generated_answer": None,
                    "query_time_seconds": None,
                    "generation_evaluation": "failed",
                    "retrieved_documents": [],
                    "retrieval_evaluation": False
                }

                logging.info(f"\nProcessing question {i+1}/{total_questions} (Source: {source_info}): {question}")
                logging.debug(f"Ground Truth Answers: {ground_truth_answers}")
                if original_context: logging.debug(f"Original Context (first 100 chars): {original_context[:100]}...")
                if document_id: logging.debug(f"Associated Document ID: {document_id}")


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
                    current_result["generated_answer"] = generated_answer
                    current_result["query_time_seconds"] = query_time

                    # --- Evaluate Generation ---
                    evaluation_result = evaluate_answer(generated_answer, ground_truth_answers)
                    evaluation_results[evaluation_result] += 1
                    logging.info(f"Generation Evaluation: {evaluation_result}")
                    current_result["generation_evaluation"] = evaluation_result

                    # --- Evaluate Retrieval (Optional, requires debug endpoint) ---
                    if DEBUG_RETRIEVER_URL:
                        retrieved_docs = get_retrieved_documents(question, DEBUG_RETRIEVER_URL)
                        current_result["retrieved_documents"] = retrieved_docs
                        
                        # Pass all relevant info for robust retrieval check
                        retrieval_successful = evaluate_retrieval(retrieved_docs, original_context, ground_truth_answers, document_id)
                        if retrieval_successful:
                            retrieval_success_count += 1
                            logging.info("Retrieval Evaluation: SUCCESS (Relevant information found in retrieved docs)")
                        else:
                            logging.warning("Retrieval Evaluation: FAILURE (Relevant information NOT found in retrieved docs)")
                        current_result["retrieval_evaluation"] = retrieval_successful
                    else:
                        logging.warning("Skipping retrieval evaluation: DEBUG_RETRIEVER_URL not provided or accessible.")

                else:
                    logging.error("Failed to get a response from the RAG service.")
                    evaluation_results["query_failed"] += 1
                
                # Append result to in-memory list (for final pretty-printed output)
                full_results_data.append(current_result)
                
                # IMMEDIATELY write result to JSONL file
                f_incremental.write(json.dumps(current_result) + '\n')
                f_incremental.flush() # Ensure it's written to disk immediately
                os.fsync(f_incremental.fileno()) # Force OS to write to disk
        
    except Exception as e:
        logging.critical(f"An unhandled error occurred during benchmarking: {e}", exc_info=True)
        # The JSONL file will still contain all results up to the point of failure.
        logging.info(f"Partial results are available in {incremental_output_filename}")
        sys.exit(1) # Exit with an error code

    # --- Report Metrics ---
    logging.info("\n--- Benchmarking Results ---")
    logging.info(f"Data Source: {args.data_source_type}")
    if args.data_source_type == "huggingface":
        logging.info(f"Dataset: {args.dataset_id}, Split: {args.split}")
    elif args.data_source_type == "local_qna":
        logging.info(f"Local Q&A Directory: {args.local_qna_dir}")

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
         logging.info(f"\nRetrieval Evaluation (Based on finding document ID, original context, or ground truth answer in retrieved docs):")
         logging.info(f"Successful Retrievals: {retrieval_success_count} ({retrieval_success_count/total_questions:.2%})")
         logging.info(f"Failed Retrievals: {total_questions - retrieval_success_count} ({(total_questions - retrieval_success_count)/total_questions:.2%})")
    else:
         logging.warning("\nRetrieval evaluation skipped.")

    # Save detailed results to a final, pretty-printed JSON file (if script completes)
    final_output_filename = f"benchmark_results_{args.data_source_type}_{timestamp}.json"
    try:
        with open(final_output_filename, 'w', encoding='utf-8') as f_final:
            json.dump(full_results_data, f_final, indent=4)
        logging.info(f"\nDetailed benchmark results saved to: {final_output_filename}")
    except Exception as e:
        logging.error(f"Error saving detailed results to final JSON file: {e}", exc_info=True)


    logging.info("\n--- Benchmarking Complete ---")
    logging.info(f"Full results in JSONL format: {incremental_output_filename}")
    logging.info(f"Summary results in JSON format (if completed): {final_output_filename}")
    logging.info("Review the logs and the generated JSON files for detailed results.")