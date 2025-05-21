# huggingface_dataset_fetcher.py
# This script demonstrates how to fetch and load a dataset from the
# Hugging Face Hub using the 'datasets' library.
# It can accept the dataset ID and split as command-line arguments.

import sys
import logging
import argparse # Import argparse for command-line arguments
from datasets import load_dataset, Dataset, DatasetDict # Import necessary classes
import os # Import os to set HF_DATASETS_CACHE

# --- Configuration ---
# Default dataset identifier on Hugging Face Hub (used if no argument is provided)
DEFAULT_DATASET_ID = "LLukas22/nq-simplified"

# Default dataset split to load (used if no argument is provided and DEFAULT_DATASET_ID is used)
# Not all datasets have the same splits. Check the dataset page on Hugging Face Hub.
# Set to None to load all available splits into a DatasetDict by default.
DEFAULT_DATASET_SPLIT = "validation" # Using 'validation' to align with our benchmark

# Default cache directory (can be overridden by HF_DATASETS_CACHE environment variable)
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ---------------------

def fetch_and_load_dataset(dataset_id, split=None):
    """
    Fetches and loads a dataset from the Hugging Face Hub.

    Args:
        dataset_id (str): The identifier of the dataset on Hugging Face Hub
                          (e.g., "squad", "LLukas22/nq-simplified").
        split (str, optional): The specific split to load (e.g., "train", "validation").
                                If None, loads all available splits into a DatasetDict.

    Returns:
        Dataset or DatasetDict: The loaded dataset object(s), or None if loading fails.
    """
    logging.info(f"Attempting to load dataset '{dataset_id}' from Hugging Face Hub.")
    if split:
        logging.info(f"Loading split: '{split}'")
    else:
        logging.info("Loading all available splits.")

    try:
        # The load_dataset function automatically handles downloading and caching
        dataset = load_dataset(dataset_id, split=split)
        logging.info(f"✅ Successfully loaded dataset '{dataset_id}'.")
        if isinstance(dataset, DatasetDict):
             logging.info(f"Loaded dataset is a DatasetDict with splits: {list(dataset.keys())}")
        elif isinstance(dataset, Dataset):
             logging.info(f"Loaded dataset is a single Dataset object.")
             logging.info(f"Number of examples in the loaded split: {len(dataset)}")

        return dataset

    except FileNotFoundError:
        logging.error(f"❌ Dataset '{dataset_id}' not found on Hugging Face Hub.")
        logging.error("Please check the dataset ID.")
        return None
    except ValueError as ve:
        logging.error(f"❌ Error loading dataset '{dataset_id}': {ve}")
        if split:
             logging.error(f"Check if the split '{split}' exists for this dataset.")
        return None
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred while loading dataset '{dataset_id}': {e}", exc_info=True)
        return None

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Hugging Face Dataset Fetcher Script ---")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Fetch and load a dataset from Hugging Face Hub.")
    parser.add_argument(
        "dataset_id",
        nargs="?", # Make the argument optional
        default=DEFAULT_DATASET_ID,
        help=f"The identifier of the dataset on Hugging Face Hub (default: {DEFAULT_DATASET_ID})"
    )
    parser.add_argument(
        "--split",
        "-s",
        default=DEFAULT_DATASET_SPLIT,
        help=f"The dataset split to load (e.g., 'train', 'validation', 'test'). Set to 'None' to load all splits (default: {DEFAULT_DATASET_SPLIT})"
    )
    parser.add_argument(
        "--cache_dir",
        "-c",
        default=os.environ.get("HF_DATASETS_CACHE", DEFAULT_CACHE_DIR), # Use env var if set, otherwise default
        help=f"The directory to cache datasets (default: {os.environ.get('HF_DATASETS_CACHE', DEFAULT_CACHE_DIR)})"
    )

    args = parser.parse_args()

    # Set the HF_DATASETS_CACHE environment variable based on the argument or default
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    logging.info(f"Hugging Face Datasets Cache Directory set to: {os.environ['HF_DATASETS_CACHE']}")

    # Handle the case where the user explicitly passes 'None' for the split
    dataset_split_to_load = None if args.split.lower() == 'none' else args.split


    logging.info(f"Fetching dataset: {args.dataset_id}")
    if dataset_split_to_load:
        logging.info(f"Fetching split: {dataset_split_to_load}")
    else:
        logging.info("Fetching all splits.")


    # Load the dataset using the provided arguments
    loaded_dataset = fetch_and_load_dataset(args.dataset_id, split=dataset_split_to_load)

    if loaded_dataset:
        logging.info("\n--- Dataset Loaded Successfully ---")
        logging.info(f"Dataset object type: {type(loaded_dataset)}")

        # --- How to access the data ---
        # If you loaded a single split (split is not None):
        # loaded_dataset is a 'datasets.Dataset' object.
        # You can access examples by index:
        if isinstance(loaded_dataset, Dataset):
             logging.info("\nExample of accessing data (first example):")
             try:
                 if len(loaded_dataset) > 0:
                      first_example = loaded_dataset[0]
                      logging.info(f"First example keys: {list(first_example.keys())}")
                      # Access specific fields, e.g., 'question', 'context', 'answers'
                      # The exact field names depend on the dataset schema.
                      # You can print the schema: print(loaded_dataset.features)
                      # For 'LLukas22/nq-simplified', common keys are 'question', 'context', 'answers'
                      if 'question' in first_example:
                           logging.info(f"  Question: {first_example['question']}")
                      if 'context' in first_example:
                           logging.info(f"  Context (first 100 chars): {first_example['context'][:100]}...")
                      if 'answers' in first_example and first_example['answers']:
                           logging.info(f"  Answers: {first_example['answers']}")
                      else:
                           logging.warning("  'question', 'context', or 'answers' keys not found in the first example.")
                           logging.info(f"  Dataset features: {loaded_dataset.features}")
                 else:
                      logging.warning("Dataset is empty, cannot access first example.")
             except IndexError:
                 logging.warning("Dataset is empty, cannot access first example.")
             except KeyError as ke:
                 logging.warning(f"Could not access expected key in first example: {ke}. Check dataset features.")
                 logging.info(f"Dataset features: {loaded_dataset.features}")


        # If you loaded all splits (split is None):
        # loaded_dataset is a 'datasets.DatasetDict' object.
        # You can access individual splits by name (e.g., 'train', 'validation'):
        elif isinstance(loaded_dataset, DatasetDict):
             logging.info("\nExample of accessing data (first example from each split):")
             for split_name, dataset_split in loaded_dataset.items():
                 logging.info(f"--- Split: {split_name} ({len(dataset_split)} examples) ---")
                 try:
                     if len(dataset_split) > 0:
                          first_example = dataset_split[0]
                          logging.info(f"  First example keys: {list(first_example.keys())}")
                          if 'question' in first_example:
                               logging.info(f"    Question: {first_example['question']}")
                          if 'context' in first_example:
                               logging.info(f"    Context (first 100 chars): {first_example['context'][:100]}...")
                          if 'answers' in first_example and first_example['answers']:
                               logging.info(f"    Answers: {first_example['answers']}")
                          else:
                               logging.warning(f"  'question', 'context', or 'answers' keys not found in the first example for split {split_name}.")
                               logging.info(f"  Dataset features for split {split_name}: {dataset_split.features}")
                     else:
                          logging.info("  Split is empty.")
                 except KeyError as ke:
                      logging.warning(f"Could not access expected key in first example for split {split_name}: {ke}. Check dataset features.")
                      logging.info(f"Dataset features for split {split_name}: {dataset_split.features}")


        logging.info("\nDataset object is now available in the 'loaded_dataset' variable.")
        logging.info("You can use this script as a starting point to integrate dataset loading into other workflows.")

    else:
        logging.error("\n--- Failed to Load Dataset ---")
        logging.error("Please check the dataset ID, split name, cache directory, and your internet connection.")

    logging.info("\n--- Script Finished ---")
