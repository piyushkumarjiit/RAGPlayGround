# indexer_service.py
# This script monitors specified data folders for changes (read from config.ini)
# and performs incremental updates to the FAISS index using LangChain methods.
# It tracks file modification times to trigger index rebuilds when changes are detected.
# It runs as a standalone background service.

import os
import time
import threading
from dotenv import load_dotenv
import datetime
import configparser
import sys
import logging
import traceback
import json
import uuid # Import the uuid module

# --- watchdog imports ---
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
# ------------------------

# LangChain/Ollama/FAISS imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import CSVLoader
# from langchain_community.document_loaders import JSONLoader # Removed JSONLoader for now
from langchain_community.document_loaders import UnstructuredExcelLoader # Added Excel Loader
from langchain_community.document_loaders import UnstructuredPowerPointLoader # Added PowerPoint Loader


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Use langchain_community for FAISS
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document


# --- Configuration ---
load_dotenv()

DEFAULT_CONFIG_FILE = "config.ini"
FAISS_PATH = "faiss_index" # Directory where FAISS index is persisted
LLM_MODEL_NAME = "qwen3:8b" # Ensure this matches the FastAPI service
EMBEDDING_MODEL_NAME = "nomic-embed-text" # MUST match the model used by the FastAPI service!
INDEX_READY_FILE = os.path.join(FAISS_PATH, "index_ready.timestamp") # File to signal index readiness
INDEX_STATE_FILE = os.path.join(FAISS_PATH, "indexed_files_state.json") # Track file mtimes


# Ensure directories exist
os.makedirs(FAISS_PATH, exist_ok=True)


# --- Global variables ---
embedding_function = None
is_indexing = False # Flag to indicate if indexing is currently in progress
MONITORED_PATHS = []
indexed_files_state = {} # Tracks file_path: mtime
# ------------------------


# --- Logging Setup ---
# Configured in main execution block

# --- State Management Functions ---

def load_index_state(state_file_path=INDEX_STATE_FILE):
    """Loads the state of indexed files (mtime) from a JSON file."""
    if not os.path.exists(state_file_path):
        logging.info(f"Index state file not found: {state_file_path}. Starting with empty state.")
        return {}
    try:
        with open(state_file_path, 'r') as f:
            state = json.load(f)
            if not isinstance(state, dict):
                 logging.error(f"Invalid state file format in {state_file_path}. Expected dictionary.")
                 return {}
            # Basic validation: ensure values are numbers (mtime)
            cleaned_state = {}
            for file_path, mtime in state.items():
                 if isinstance(mtime, (int, float)):
                      cleaned_state[file_path] = mtime
                 else:
                      logging.warning(f"Invalid mtime format for file {file_path} in {state_file_path}. Skipping entry.")
                      continue
            logging.info(f"Loaded index state from: {state_file_path} ({len(cleaned_state)} files tracked)")
            return cleaned_state
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from index state file {state_file_path}.", exc_info=True)
        logging.warning("Starting with empty index state due to JSON error.")
        return {}
    except Exception as e:
        logging.error(f"Error loading index state from {state_file_path}: {e}", exc_info=True)
        logging.warning("Starting with empty index state due to load error.")
        return {}

def save_index_state(state, state_file_path=INDEX_STATE_FILE):
    """Saves the current state of indexed files (mtime) to a JSON file."""
    try:
        os.makedirs(os.path.dirname(state_file_path), exist_ok=True)
        with open(state_file_path, 'w') as f:
            json.dump(state, f, indent=4)
        logging.info(f"Saved index state to: {state_file_path} ({len(state)} files tracked)")
    except Exception as e:
        logging.error(f"Error saving index state to {state_file_path}: {e}", exc_info=True)


# --- Document Loading ---

def load_document_with_path(file_path):
    """Loads a single document and returns it with its file path."""
    try:
        # Use a simple case-insensitive check for file extensions
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension in (".md", ".txt"):
            loader = TextLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif file_extension == ".csv":
            loader = CSVLoader(file_path)
        elif file_extension in (".xlsx", ".xls"): # Added Excel extensions
            logging.debug(f"Using UnstructuredExcelLoader for file: {file_path}")
            loader = UnstructuredExcelLoader(file_path)
        elif file_extension in (".pptx", ".ppt"): # Added PowerPoint extensions
            logging.debug(f"Using UnstructuredPowerPointLoader for file: {file_path}")
            loader = UnstructuredPowerPointLoader(file_path)
        # Add more loaders here for other formats
        else:
            logging.debug(f"Skipping unsupported file format: {file_path}")
            return None # Return None for unsupported types

        docs = loader.load()
        # Add source metadata to each document
        for doc in docs:
            doc.metadata['source'] = os.path.abspath(file_path) # Add the original file path as source (use abspath for consistency)
            # Optionally add mtime to metadata as well? doc.metadata['mtime'] = os.path.getmtime(file_path)
        logging.debug(f"Loaded {len(docs)} pages/sections from {file_path}")
        return docs

    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}", exc_info=True)
        return None


def get_current_files_state(paths_to_scan):
    """Scans the monitored paths and returns the current state of files (path: mtime)."""
    current_state = {}
    logging.info(f"Scanning paths for current file state: {paths_to_scan}")
    if not paths_to_scan:
        logging.warning("No paths specified to scan.")
        return {}

    supported_extensions = (".pdf", ".md", ".txt", ".docx", ".csv", ".xlsx", ".xls", ".pptx", ".ppt") # Added Excel and PowerPoint extensions

    for data_path in paths_to_scan:
        if not os.path.exists(data_path):
            logging.warning(f"Scan path does not exist: {data_path}. Skipping.")
            continue

        for root, _, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Ignore directories and hidden files (basic check)
                if not os.path.isfile(file_path) or file.startswith('.'):
                     continue
                # Check if the file extension is one we support
                if not file_path.lower().endswith(supported_extensions):
                     continue # Skip unsupported file types

                try:
                    mtime = os.path.getmtime(os.path.abspath(file_path)) # Use abspath for consistency
                    current_state[os.path.abspath(file_path)] = mtime # Use abspath for consistency
                except Exception as e:
                    logging.error(f"Error getting mtime for file {file_path}: {e}", exc_info=True)

    logging.info(f"Scanned {len(current_state)} files.")
    return current_state


# Original split_documents function - no changes needed
def split_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(all_splits)} chunks")
    return all_splits

# Original get_embedding_function - no changes needed
def get_embedding_function(model_name=EMBEDDING_MODEL_NAME):
    """Initializes the Ollama embedding function."""
    logging.info(f"Initializing Ollama embeddings with model: {model_name}")
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        embeddings.embed_query("test") # Test if it works
        logging.info(f"Initialized Ollama embeddings successfully.")
        return embeddings
    except Exception as e:
         logging.error(f"Error initializing Ollama embeddings: {e}", exc_info=True)
         logging.error(f"Please ensure 'ollama serve' is running and the model '{model_name}' is pulled.")
         raise


# --- Indexing Logic (Rebuilds index on changes) ---

def update_index_incrementally():
    """
    Performs incremental updates to the FAISS index by rebuilding it
    from the current set of files in monitored directories when changes are detected.
    """
    global is_indexing, embedding_function, MONITORED_PATHS, indexed_files_state

    if is_indexing:
        logging.debug("Indexing is already in progress. Skipping this update request.")
        return

    is_indexing = True
    logging.info("\n--- Initiating Incremental Index Update (FAISS Rebuild) ---")

    try:
        if embedding_function is None:
             logging.warning("Embedding function not initialized! Attempting to initialize now...")
             embedding_function = get_embedding_function()
             if embedding_function is None:
                  logging.error("Embedding function failed to initialize. Cannot proceed with indexing.")
                  return # Exit if embeddings are not available


        last_indexed_state = indexed_files_state.copy()
        current_files_state = get_current_files_state(MONITORED_PATHS)

        files_changed = False

        # Determine if any files have been added, deleted, or modified
        if set(last_indexed_state.keys()) != set(current_files_state.keys()):
             files_changed = True
             logging.info("File list has changed (added or deleted files).")
        else:
             for file_path, current_mtime in current_files_state.items():
                  if last_indexed_state.get(file_path) != current_mtime:
                       files_changed = True
                       logging.info(f"File modified: {file_path}")
                       break # Found a modified file, no need to check others

        if files_changed or not last_indexed_state: # Rebuild if changes or if starting with empty state
            logging.info("Changes detected or starting fresh. Rebuilding FAISS index from all current files...")

            all_chunks_to_index = []
            new_indexed_files_state = {}

            # Load and split all current documents
            for file_path, current_mtime in current_files_state.items():
                 docs = load_document_with_path(file_path)
                 if docs:
                     chunks = split_documents(docs)
                     if chunks:
                         all_chunks_to_index.extend(chunks)
                         new_indexed_files_state[file_path] = current_mtime
                     else:
                         logging.warning(f"No chunks generated for file: {file_path}")
                 else:
                      logging.warning(f"No documents loaded for file: {file_path}")


            if all_chunks_to_index:
                 logging.info(f"Creating/Rebuilding FAISS index with {len(all_chunks_to_index)} chunks.")
                 try:
                     # Create a new FAISS index from the current chunks
                     new_vector_store = FAISS.from_documents(all_chunks_to_index, embedding_function)

                     # Save the new index locally
                     new_vector_store.save_local(FAISS_PATH)
                     logging.info(f"FAISS index rebuilt and saved to {FAISS_PATH}.")

                     # Update the global state variable
                     indexed_files_state = new_indexed_files_state

                     # Save State File
                     save_index_state(indexed_files_state) # Save the updated state to disk

                     # Signal readiness
                     signal_index_ready()

                 except Exception as e:
                      logging.error(f"Error creating/rebuilding FAISS index: {e}", exc_info=True)
                      logging.warning("FAISS index rebuild failed. Index state NOT saved.")
                      # Decide how to handle indexing errors - for now, log and continue


            else:
                 logging.warning("No documents found or processed to build the FAISS index.")
                 # If no documents are found but there was a previous state, clear the index files and state
                 if last_indexed_state:
                      logging.info("Clearing old FAISS index files and state as no documents are present.")
                      try:
                           if os.path.exists(FAISS_PATH):
                                for item in os.listdir(FAISS_PATH):
                                     item_path = os.path.join(FAISS_PATH, item)
                                     if os.path.isfile(item_path):
                                          os.remove(item_path)
                           indexed_files_state = {}
                           save_index_state(indexed_files_state)
                           signal_index_ready() # Signal readiness even if empty
                      except Exception as e:
                           logging.error(f"Error clearing FAISS index files: {e}", exc_info=True)


        else:
            logging.info("No changes detected. FAISS index is up-to-date.")


        logging.info("--- Incremental Index Update Complete ---")

    except Exception as e:
        # This catches errors during state loading, scanning, or unhandled indexing errors
        logging.error(f"\n--- Error during Incremental Index Update: {e} ---", exc_info=True)
    finally:
        is_indexing = False


# --- Watchdog Setup ---

class DocumentEventHandler(FileSystemEventHandler):
    """Handles file system events in the data directory."""
    def __init__(self, monitored_paths, ignore_paths=None):
        super().__init__()
        self.monitored_paths = [os.path.abspath(p) for p in monitored_paths]
        self.ignore_paths = [os.path.abspath(p) for p in ignore_paths] if ignore_paths else []
        # Add FAISS index files to ignore list
        self.ignore_paths.append(os.path.abspath(FAISS_PATH))


        self._timer = None
        self._lock = threading.Lock()

        logging.info(f"Watchdog set up to monitor: {self.monitored_paths}")
        if self.ignore_paths:
            logging.info(f"Watchdog ignoring paths starting with: {self.ignore_paths}")


    def _should_ignore(self, event_path):
        """Checks if the event path should be ignored."""
        abs_path = os.path.abspath(event_path)

        if "~$" in event_path or event_path.endswith((".tmp", ".temp", ".swp")):
             logging.debug(f"WATCHDOG: Ignoring temporary file pattern: {event_path}")
             return True

        for ignored_path in self.ignore_paths:
             if abs_path.startswith(ignored_path):
                 logging.debug(f"WATCHDOG: Ignoring event in ignored path: {event_path}")
                 return True

        is_within_monitored_path = False
        for monitored_path in self.monitored_paths:
             if abs_path.startswith(monitored_path):
                 is_within_monitored_path = True
                 break
        if not is_within_monitored_path:
             logging.debug(f"WATCHDOG: Ignoring event outside monitored paths: {event_path}")
             return True


        return False

    def on_any_event(self, event):
        if event.is_directory:
             if event.event_type == 'created':
                  is_within_monitored_path = False
                  for monitored_path in self.monitored_paths:
                       if os.path.abspath(event.src_path).startswith(monitored_path):
                            is_within_monitored_path = True
                            break
                  if not is_within_monitored_path:
                       logging.debug(f"WATCHDOG: Ignoring directory creation outside monitored paths: {event.src_path}")
                       return
                  pass
             else:
                  if not (event.event_type == 'moved' and hasattr(event, 'dest_path') and os.path.abspath(event.dest_path).startswith(self.monitored_paths[0] if self.monitored_paths else '')) :
                       logging.debug(f"WATCHDOG: Ignoring non-creation directory event: {event.event_type} on {event.src_path}")
                       return


        if self._should_ignore(event.src_path):
            return

        logging.info(f"WATCHDOG: Detected RELEVANT file system event: {event.event_type} on {event.src_path}")

        debounce_delay_seconds = 2.0
        with self._lock:
            if self._timer:
                self._timer.cancel()
                logging.debug(f"WATCHDOG: Debounce timer cancelled for {event.src_path}")
            logging.debug(f"WATCHDOG: Starting debounce timer for {event.src_path}")
            self._timer = threading.Timer(debounce_delay_seconds, update_index_incrementally)
            self._timer.start()
            logging.info(f"WATCHDOG: Scheduled index update in {debounce_delay_seconds} seconds.")


observer = None


# --- Configuration Loading ---

def load_config(config_file_path=DEFAULT_CONFIG_FILE):
    """Loads configuration from the specified INI file."""
    config = configparser.ConfigParser()
    logging.info(f"Attempting to load configuration from: {config_file_path}")
    try:
        config.read(config_file_path, encoding='utf-8')
        logging.info("Configuration loaded.")
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file_path}")
        return None
    except configparser.Error as e:
        logging.error(f"Error reading configuration file {config_file_path}: {e}", exc_info=True)
        return None
    return config

def setup_logging(config):
    """Confgures the logging system based on the config."""
    log_level_str = config.get('Logging', 'level', fallback='INFO').upper()
    log_file = config.get('Logging', 'log_file', fallback='indexer_service.log')
    log_format = config.get('Logging', 'format', fallback='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log_level = getattr(logging, log_level_str, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Set root logger to DEBUG to capture all messages

    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    try:
        resolved_log_file = os.path.abspath(log_file)
        file_handler = logging.FileHandler(resolved_log_file)
        file_handler.setLevel(log_level) # Set file handler level from config
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)

        root_logger.addHandler(file_handler)
        logging.info(f"Logging configured to file: {resolved_log_file} with level: {log_level_str}")

        # Add a console handler for visibility during development
        console_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout for console output
        console_handler.setLevel(logging.INFO) # Set console level to INFO by default
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # Simpler format for console
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        logging.info("Console logging enabled with level: INFO")


    except Exception as e:
        print(f"ERROR: Error setting up file logging to {resolved_log_file}: {e}")
        logging.warning("File logging setup failed. Logging will only be to console.")


def get_monitored_paths_from_config(config):
    """Extracts the list of monitored paths from the config object."""
    paths = []
    if config and 'Indexer' in config and 'monitor_paths' in config['Indexer']:
        paths_str = config['Indexer']['monitor_paths']
        paths = [line.strip() for line in paths_str.splitlines() if line.strip()]
        logging.info(f"Read {len(paths)} monitored paths from config: {paths}")
    else:
        logging.warning("Warning: 'Indexer' section or 'monitor_paths' key not found in config.")
        logging.warning("Defaulting to monitoring the './data' folder.")
        paths = ["./data"]

    if paths == ["./data"]:
         try:
             os.makedirs("./data", exist_ok=True)
             logging.info("Ensured default './data' directory exists.")
         except Exception as e:
             logging.error(f"Error creating default './data' directory: {e}", exc_info=True)


    absolute_paths = [os.path.abspath(p) for p in paths]
    logging.debug(f"Resolved monitored paths to absolute: {absolute_paths}")
    return absolute_paths


# --- Signal Index Ready ---
def signal_index_ready():
    """Creates or updates a timestamp file to signal that the index is ready."""
    try:
        os.makedirs(os.path.dirname(INDEX_READY_FILE), exist_ok=True)
        with open(INDEX_READY_FILE, 'w') as f:
            f.write(str(time.time())) # Use time.time() for a numerical timestamp
        logging.info(f"Signaled index readiness by updating: {INDEX_READY_FILE}")
    except Exception as e:
        logging.error(f"Error signaling index readiness: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Indexer Service starting from: {os.getcwd()}")

    config_file = DEFAULT_CONFIG_FILE
    if len(sys.argv) > 1:
         if sys.argv[1].lower().endswith(".ini"):
              config_file = sys.argv[1]
         else:
              print(f"Warning: Ignoring unexpected command line argument: {sys.argv[1]}")

    config = load_config(config_file)

    if config is None:
         print(f"FATAL: Could not load configuration from {config_file}. Exiting.")
         sys.exit(1)

    setup_logging(config)
    logging.info("--- Initializing Indexer Service ---")
    logging.info(f"Indexer Service running from: {os.getcwd()}")

    try:
        MONITORED_PATHS = get_monitored_paths_from_config(config)

        if not MONITORED_PATHS:
             logging.critical("No valid paths to monitor specified in configuration. Exiting.")
             sys.exit(1)

        embedding_function = get_embedding_function(EMBEDDING_MODEL_NAME)

        # Load initial state and perform initial index update (rebuild)
        indexed_files_state = load_index_state()
        update_index_incrementally() # Perform initial index build/rebuild


        logging.info(f"\n--- Starting watchdog observer for {MONITORED_PATHS} ---")
        # Ignore the FAISS_PATH directory itself
        event_handler = DocumentEventHandler(MONITORED_PATHS, ignore_paths=[os.path.abspath(FAISS_PATH)])
        observer = Observer()
        scheduled_paths = []
        for path in MONITORED_PATHS:
             abs_path = os.path.abspath(path)
             if os.path.exists(abs_path):
                 observer.schedule(event_handler, abs_path, recursive=True)
                 scheduled_paths.append(abs_path)
                 logging.info(f" - Scheduled monitoring for: {abs_path}")
             else:
                 logging.warning(f"Configured path does not exist, skipping monitoring: {abs_path}")

        if not scheduled_paths:
             logging.critical("No valid existing paths scheduled for monitoring. Exiting.")
             sys.exit(1)


        observer.start()
        logging.info("Watchdog observer started. Monitoring file changes...")
        logging.info("Indexer Service is running. Press Ctrl+C to stop.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("\nCtrl+C detected. Stopping Indexer Service.")
    except Exception as e:
         logging.critical(f"An unexpected fatal error occurred during Indexer Service execution: {e}", exc_info=True)
    finally:
        if observer:
            logging.info("Stopping watchdog observer...")
            observer.stop()
            observer.join()
            logging.info("Watchdog observer stopped.")
        logging.info("Indexer Service finished.")
