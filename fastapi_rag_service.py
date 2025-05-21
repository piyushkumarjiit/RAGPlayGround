# fastapi_rag_service.py
# This script runs a FastAPI web service that answers RAG queries
# using a FAISS index managed by the separate indexer_service.py.
# It loads the index on startup, moves it to the GPU, and periodically
# checks for updates signaled by the indexer_service to reload and re-move
# the index to the GPU. It includes a debug endpoint for the retriever
# and implements timeouts for Ollama calls by running them in a thread pool.

import os
import time
import threading
from dotenv import load_dotenv
import datetime
import asyncio
import uvicorn
import logging # Import logging for this service too
import re # Import regex module
import concurrent.futures # Import for ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from contextlib import asynccontextmanager

# LangChain/FAISS/Ollama imports
# Use langchain_community for FAISS
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage # Import BaseMessage

# Import FAISS GPU specific components
import faiss
from faiss import StandardGpuResources, index_cpu_to_gpu


# --- Configuration ---
load_dotenv()

DEFAULT_CONFIG_FILE = "config.ini" # Assuming config.ini is in the same directory
FAISS_PATH = "faiss_index" # Directory where FAISS index is persisted
LLM_MODEL_NAME = "qwen3:8b" # Ensure this matches the FastAPI service
EMBEDDING_MODEL_NAME = "nomic-embed-text" # MUST match the model used by the indexer!
INDEX_READY_FILE = os.path.join(FAISS_PATH, "index_ready.timestamp") # File to signal index readiness
INDEX_CHECK_INTERVAL_SECONDS = 10 # Reduced check interval for faster testing

# --- GPU Configuration ---
# You can specify which GPU device to use (0 is typically the first GPU)
GPU_DEVICE_ID = 0

# --- Ollama Timeout Configuration ---
# Define a global variable for the timeout in seconds
# Adjust this value based on your expected maximum processing time for Ollama requests
OLLAMA_REQUEST_TIMEOUT_SECONDS = 60 # Set a default timeout of 60 seconds

# --- Global variables to hold RAG components ---
embedding_function = None
vector_store = None # FAISS vector store instance (will hold the GPU index)
llm_instance = None
prompt_template = None
output_parser_instance = StrOutputParser()
last_index_timestamp = None # To track the last time we loaded the index
gpu_resources = None # Global variable to hold GPU resources

# --- Thread Pool Executor for Blocking Calls ---
# Use a ThreadPoolExecutor for running synchronous/blocking code (like Ollama calls)
# Adjust max_workers based on your system's capabilities and expected load
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5) # Example: 5 worker threads


# --- Lock for synchronizing access to global RAG components ---
reload_lock = threading.Lock()
# -------------------------------------------------------------


# --- Logging Setup for FastAPI Service ---
# Configure logging to a file and the console
logging.basicConfig(
    level=logging.DEBUG, # Set minimum level to DEBUG to see all messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='fastapi_rag_service.log', # Specify the log file name
    filemode='a' # Append to the log file if it exists
)

# Optional: Add a console handler if you still want logs in the terminal
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO) # Set level for console output
# console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# logging.getLogger().addHandler(console_handler)
# -----------------------------------------

# --- Custom Output Processing ---

# Modified to accept BaseMessage or string and extract content
def remove_think_block(output: BaseMessage | str) -> str:
    """
    Removes the <think>...</think> block from the LLM's output.
    Accepts a BaseMessage or string and extracts the string content.
    """
    if isinstance(output, BaseMessage):
        text = output.content # Extract string content from BaseMessage
    elif isinstance(output, str):
        text = output # Use string directly if already a string
    else:
        logging.warning(f"remove_think_block received unexpected input type: {type(output)}")
        return str(output) # Return string representation for unexpected types

    # Use regex to find and remove the block, including the tags
    # The pattern <think>.*?</think> matches the tags and any content between them (non-greedy)
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL) # re.DOTALL allows '.' to match newlines
    return cleaned_text.strip() # Remove leading/trailing whitespace that might be left


# --- RAG Pipeline Helper Functions ---

def get_embedding_function(model_name=EMBEDDING_MODEL_NAME):
    """Initializes the Ollama embedding function (must match indexer)."""
    logging.info(f"Initializing Ollama embeddings with model: {model_name} (for FAISS)")
    try:
        logging.debug(f"Attempting to create OllamaEmbeddings instance for model: {model_name}")
        embeddings = OllamaEmbeddings(model=model_name)
        logging.debug("OllamaEmbeddings instance created.")
        logging.info(f"Initialized Ollama embeddings successfully.")
        return embeddings
    except Exception as e:
         logging.error(f"Error initializing Ollama embeddings: {e}", exc_info=True)
         logging.error(f"Please ensure 'ollama serve' is running and the model '{model_name}' is pulled.")
         # Depending on criticality, you might want to raise the exception or return None
         # For now, let's log and return None, allowing the service to start but not query
         return None # Return None if embedding function initialization fails

def load_vector_store(embedding_function, gpu_resources, persist_directory=FAISS_PATH, gpu_device_id=GPU_DEVICE_ID):
    """
    Loads an existing FAISS vector store from disk (onto CPU) and then moves it to the GPU.
    """
    logging.info(f"Attempting to load FAISS vector store from: {persist_directory}")
    if embedding_function is None:
         logging.error("Embedding function is None, cannot load vector store.")
         return None
    if gpu_resources is None:
         logging.error("GPU resources not initialized, cannot move index to GPU.")
         return None


    # Check if the FAISS index files exist
    index_file = os.path.join(persist_directory, "index.faiss")
    pkl_file = os.path.join(persist_directory, "index.pkl")

    if not os.path.exists(index_file) or not os.path.exists(pkl_file):
        logging.warning(f"FAISS index files not found in {persist_directory}. Cannot load vector store.")
        return None

    try:
        # Load the index onto the CPU first
        logging.info(f"Loading FAISS index onto CPU from {persist_directory}...")
        vectorstore_cpu = FAISS.load_local(
            persist_directory,
            embedding_function, # Pass the full embedding function object
            allow_dangerous_deserialization=True # Required for loading FAISS indexes
        )
        logging.info("FAISS index loaded onto CPU.")

        # Move the index from CPU to GPU
        logging.info(f"Attempting to move FAISS index to GPU device {gpu_device_id}...")
        try:
            # Access the underlying FAISS index object from the LangChain wrapper
            index_cpu_faiss = vectorstore_cpu.index
            # Move the raw FAISS index to the GPU
            index_gpu_faiss = index_cpu_to_gpu(gpu_resources, gpu_device_id, index_cpu_faiss)
            logging.info(f"✅ Successfully moved FAISS index to GPU device {gpu_device_id}. Index has {index_gpu_faiss.ntotal} vectors.")

            # Create a new LangChain FAISS wrapper using the GPU index
            # Pass the full embedding function object here as well
            vectorstore_gpu = FAISS(embedding_function, index_gpu_faiss, vectorstore_cpu.docstore, vectorstore_cpu.index_to_docstore_id)

            logging.info(f"FAISS GPU vector store loaded and moved to GPU successfully.")
            return vectorstore_gpu


        except Exception as gpu_e:
             logging.error(f"❌ Failed to move FAISS index to GPU device {gpu_device_id}: {gpu_e}", exc_info=True)
             logging.error("Ensure your FAISS build correctly enabled GPU support and your CUDA/driver setup is functional.")
             # Return None if moving to GPU fails
             return None


    except Exception as e:
        logging.error(f"Error loading FAISS vector store from {persist_directory}: {e}", exc_info=True)
        return None


def initialize_rag_components(llm_model_name=LLM_MODEL_NAME, context_window=8192):
    """Initializes global LLM, Prompt, and OutputParser instances."""
    global llm_instance, prompt_template, output_parser_instance
    logging.info("Initializing global RAG chain components (LLM, Prompt)...")
    logging.debug(f"Attempting to initialize ChatOllama with model: {llm_model_name}")
    try:
        llm_instance = ChatOllama(
            model=llm_model_name,
            temperature=0,
            num_ctx=context_window
        )
        logging.debug("ChatOllama initialization successful.")
        logging.info(f"Initialized ChatOllama with model: {llm_model_name}, context window: {context_window}")

        template = """Based on the following context, identify and list the job duties for the person mentioned in the question.
If the person is not mentioned in the context or their duties are not listed, state that the information is not available in the provided context.

Context:
{context}

Question: {question}
"""
        prompt_template = ChatPromptTemplate.from_template(template)
        logging.info("Refined prompt template created.")

        # Output parser is already initialized globally

        logging.info("Global RAG chain components initialized.")
        return True # Indicate success
    except Exception as e:
        logging.error(f"Error initializing global RAG chain components: {e}", exc_info=True)
        llm_instance = None # Reset on failure
        prompt_template = None # Reset on failure
        return False # Indicate failure


# --- Index Monitoring Background Task ---

async def monitor_index_updates():
    """Periodically checks the index_ready.timestamp file for updates and reloads the index."""
    global vector_store, last_index_timestamp, embedding_function, reload_lock, gpu_resources

    logging.info(f"Starting index monitor task. Checking every {INDEX_CHECK_INTERVAL_SECONDS} seconds.")

    while True:
        # Log that the check is happening
        logging.debug(f"Monitor task: Checking for index updates at {datetime.datetime.now()}")

        await asyncio.sleep(INDEX_CHECK_INTERVAL_SECONDS)

        # Check if the timestamp file exists before trying to get its mtime
        if not os.path.exists(INDEX_READY_FILE):
            logging.debug(f"Monitor task: Index ready file not found: {INDEX_READY_FILE}. Skipping update check.")
            continue

        try:
            # Use os.path.getmtime for the last modification time
            current_timestamp = os.path.getmtime(INDEX_READY_FILE)
            logging.debug(f"Monitor task: Current index timestamp: {current_timestamp}, Last loaded timestamp: {last_index_timestamp}")

            # Check if the timestamp file's modification time is newer than the last time we loaded
            if last_index_timestamp is None or current_timestamp > last_index_timestamp:
                logging.info("\n--- Detected Index Update. Attempting to reload FAISS ---")

                # Acquire the lock before modifying global RAG components
                with reload_lock:
                    logging.debug("Monitor task: Acquired reload lock for reload.")
                    try:
                        # Log the object ID of the current vector_store before setting to None
                        if vector_store:
                             logging.debug(f"Monitor task: Current vector_store object ID before reload: {id(vector_store)}")
                        else:
                             logging.debug("Monitor task: Current vector_store is None before reload.")

                        # Explicitly set global vector_store to None before attempting to reload
                        logging.debug("Monitor task: Setting global vector_store to None.")
                        vector_store = None

                        # Optional: Add a small delay to allow potential cleanup (heuristic)
                        # time.sleep(1) # Uncomment for testing if a delay helps

                        # Ensure embedding function and GPU resources are initialized before loading vector store
                        # GPU resources should already be initialized in lifespan, but double-check
                        if gpu_resources is None:
                             logging.warning("Monitor task: GPU resources not initialized, attempting initialization...")
                             logging.debug("Monitor task: Attempting to initialize StandardGpuResources()...")
                             try:
                                 gpu_resources = StandardGpuResources()
                                 logging.debug("Monitor task: StandardGpuResources() initialized successfully.")
                                 logging.info("Monitor task: GPU resources initialized.")
                             except Exception as gpu_res_e:
                                 logging.error(f"Monitor task: Failed to initialize GPU resources during reload attempt: {gpu_res_e}", exc_info=True)
                                 logging.error("Monitor task: Cannot reload index onto GPU.")
                                 # Do not update timestamp if GPU resources fail
                                 continue # Skip the rest of the reload process


                        if embedding_function is None:
                             logging.warning("Monitor task: Embedding function not initialized, initializing now for reload.")
                             embedding_function = get_embedding_function()
                             if embedding_function is None:
                                  logging.error("Monitor task: Failed to initialize embedding function during reload attempt. Cannot reload index.")
                                  # Do not update timestamp if reload failed
                                  continue # Skip the rest of the reload process


                        # Load the vector store (which now includes moving to GPU)
                        new_vector_store = load_vector_store(embedding_function, gpu_resources)

                        if new_vector_store:
                            # Log the object ID of the new vector_store
                            logging.debug(f"Monitor task: New vector_store object ID after successful load: {id(new_vector_store)}")
                            # Update the global vector_store
                            vector_store = new_vector_store
                            # Only update the timestamp if the vector store was successfully loaded
                            last_index_timestamp = current_timestamp
                            logging.info("--- FAISS Reload Complete. Vector store updated. ---")

                        else:
                            logging.error("Monitor task: --- Failed to load updated FAISS. Keeping previous version (which is None). ---")
                            # Do not update timestamp if vector store loading failed

                    except Exception as e:
                         logging.error(f"Monitor task: Error during index monitoring or reload within lock: {e}", exc_info=True)
                         # Do not update timestamp if an error occurred during reload
                    finally:
                         # Lock is automatically released by 'with' statement
                         logging.debug("Monitor task: Released reload lock.")
            else:
                 logging.debug("Monitor task: Index ready timestamp is not newer than last loaded timestamp.")

        except Exception as e:
            # This catches errors outside the lock, like file access issues for the timestamp file
            logging.error(f"Monitor task: Error during index monitoring (outside lock): {e}", exc_info=True)


# --- FastAPI Application Setup ---

# Use asynccontextmanager for newer FastAPI versions (>=0.95.0)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Logic
    logging.info("FastAPI RAG Service starting up...")
    global embedding_function, vector_store, last_index_timestamp, reload_lock, gpu_resources, executor

    try:
        # Initialize GPU resources first
        logging.info("Initializing GPU resources...")
        logging.debug("Startup: Attempting to initialize StandardGpuResources()...")
        try:
            gpu_resources = StandardGpuResources()
            logging.debug("Startup: StandardGpuResources() initialized successfully.")
            logging.info("✅ GPU resources initialized.")
        except Exception as e:
            logging.critical(f"FATAL: Failed to initialize GPU resources: {e}", exc_info=True)
            logging.critical("Ensure your FAISS build correctly enabled GPU support and your CUDA/driver setup is functional.")
            # Depending on criticality, you might want to raise an exception here
            # For now, log critical and allow service to start, but it won't use GPU.
            gpu_resources = None # Ensure it's None on failure
            pass # Continue startup even if GPU resources fail, but RAG won't use GPU
        logging.debug("Startup: GPU resources initialization attempt complete.")


        # Initialize embedding function next
        logging.debug("Startup: Attempting to initialize embedding function...")
        embedding_function = get_embedding_function()
        if embedding_function is None:
             logging.critical("FATAL: Embedding function failed to initialize. Cannot start service.")
             # Depending on how critical this is, you might want to raise an exception here
             # For now, we'll log critical and let the service start but it won't work.
             pass # Continue startup even if embeddings fail, but RAG won't work
        logging.debug("Startup: Embedding function initialization attempt complete.")


        # Initialize global RAG chain components (LLM, Prompt)
        logging.debug("Startup: Attempting to initialize RAG chain components (LLM, Prompt)...")
        if not initialize_rag_components():
             logging.critical("FATAL: Failed to initialize RAG chain components. Cannot start service.")
             pass # Continue startup, but RAG won't work
        logging.debug("Startup: RAG chain components initialization attempt complete.")


        # Acquire lock for initial load of global components
        with reload_lock:
            logging.debug("Startup: Acquired reload lock for startup.")
            try:
                # Log the object ID of the current vector_store before initial load
                if vector_store:
                     logging.debug(f"Startup: Current vector_store object ID before initial load: {id(vector_store)}")
                else:
                     logging.debug("Startup: Current vector_store is None before initial load.")

                # Load the vector store initially (includes moving to GPU if gpu_resources are available)
                logging.debug("Startup: Attempting to load vector store...")
                if embedding_function and gpu_resources: # Only attempt to load if embeddings initialized and GPU resources available
                     vector_store = load_vector_store(embedding_function, gpu_resources)
                else:
                     logging.warning("Startup: Skipping vector store load because embedding function or GPU resources are not initialized.")

                logging.debug("Startup: Vector store load attempt complete.")

                if vector_store is None:
                     logging.warning("Warning: Initial FAISS DB not found or failed to load/move to GPU. The API will not be able to answer questions until the indexer service creates/updates the DB and it can be loaded onto GPU.")
                else:
                     # Log the object ID of the initially loaded vector_store
                     logging.debug(f"Startup: Initial vector_store object ID after successful load: {id(vector_store)}")
                     # Record the initial index timestamp if the file exists and vector store loaded
                     if os.path.exists(INDEX_READY_FILE):
                          last_index_timestamp = os.path.getmtime(INDEX_READY_FILE)
                          logging.info(f"Startup: Initial index timestamp recorded: {last_index_timestamp}")
                     else:
                          logging.warning("Startup: Index ready file not found during initial load. Timestamp not recorded.")

            except Exception as e:
                 logging.critical(f"Startup: FATAL: Unexpected error during initial load within lock: {e}", exc_info=True)
            finally:
                 # Lock is automatically released by 'with' statement
                 logging.debug("Startup: Released reload lock for startup.")


        logging.info("Starting background index monitor task...")
        # Create the background task only if the embedding function and GPU resources were initialized successfully
        if embedding_function and gpu_resources:
             asyncio.create_task(monitor_index_updates())
             logging.info("Background index monitor task started.")
        else:
             logging.warning("Skipping background index monitor task due to embedding function or GPU resources initialization failure.")


        logging.info("FastAPI RAG Service startup complete.")

    except Exception as e:
        logging.critical(f"FATAL: Unexpected error during FastAPI RAG Service startup (outside lock): {e}", exc_info=True)
        # Depending on the error, you might want to re-raise or sys.exit(1)
        # For now, log and allow the service to potentially start in a non-functional state
        pass


    yield # Application is running

    # Shutdown Logic
    logging.info("FastAPI RAG Service shutting down...")
    # Shutdown the ThreadPoolExecutor
    executor.shutdown(wait=True)
    logging.info("ThreadPoolExecutor shut down.")
    logging.info("FastAPI RAG Service shut down.")


# Initialize FastAPI app with lifespan management
app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# --- Synchronous function to perform the core RAG process ---
# This function will be run in the ThreadPoolExecutor
def perform_rag_query(query: str, current_vector_store, current_llm_instance, current_prompt_template, current_embedding_function):
    """
    Synchronous function to perform the RAG query process:
    1. Retrieve documents from the vector store.
    2. Format context.
    3. Invoke the LLM chain.
    This function is designed to be run within a ThreadPoolExecutor.
    """
    logging.debug(f"ThreadPool: Starting RAG query processing for: {query}")

    # Use the FAISS retriever directly (now using the GPU index)
    retriever_instance = current_vector_store.as_retriever(search_kwargs={'k': 20}) # Increased k to 20

    # Retriever invocation (includes embedding call to Ollama)
    # This part can be blocking
    logging.debug("ThreadPool: Performing retrieval.")
    retrieved_docs = retriever_instance.invoke(query)
    logging.debug(f"ThreadPool: Retrieval returned {len(retrieved_docs)} documents.")

    # Manually format the context for the prompt
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    logging.debug(f"ThreadPool: Formatted context for prompt.") # Avoid logging full context in debug

    # Prepare the input dictionary for the prompt
    prompt_input = {"context": context_text, "question": query}

    # Assemble and Invoke the rest of the chain (LLM call)
    # This part can also be blocking
    logging.debug("ThreadPool: Invoking LLM chain.")
    rag_chain_after_retrieval = (
         current_prompt_template
         | current_llm_instance
         | RunnableLambda(remove_think_block)
         | output_parser_instance # output_parser_instance is safe to access globally
    )

    response = rag_chain_after_retrieval.invoke(prompt_input)
    logging.debug("ThreadPool: LLM chain invocation complete.")

    return response


# --- Existing Ask Endpoint (Uses global components with lock and ThreadPoolExecutor) ---
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Endpoint to ask a question to the RAG pipeline.
    Uses the globally managed LLM and Prompt components.
    Uses FAISS for retrieval and runs the core logic in a ThreadPoolExecutor with timeout.
    """
    logging.info(f"Received query: {request.query}")

    # Acquire lock before accessing global components to pass to the thread
    # The lock is released before submitting to the executor
    with reload_lock:
        logging.debug("Ask Endpoint: Acquired reload lock to access global components.")
        if vector_store is None or llm_instance is None or prompt_template is None or embedding_function is None:
            logging.warning("Ask Endpoint: RAG components not initialized during ask request.")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG service is not ready. Index or RAG components not loaded.")

        # Pass the current state of global components to the synchronous function
        # This ensures the thread uses the components that were active when the request arrived
        current_vector_store = vector_store
        current_llm_instance = llm_instance
        current_prompt_template = prompt_template
        current_embedding_function = embedding_function # Pass embedding function too, though not directly used in perform_rag_query

        logging.debug("Ask Endpoint: Released reload lock after accessing global components.")

    # Submit the synchronous RAG query function to the thread pool executor
    # Use asyncio.wrap_future to make the Future awaitable
    loop = asyncio.get_running_loop()
    rag_future = loop.run_in_executor(
        executor,
        perform_rag_query,
        request.query,
        current_vector_store,
        current_llm_instance,
        current_prompt_template,
        current_embedding_function # Pass embedding function
    )

    try:
        # Wait for the future to complete with a timeout
        logging.debug(f"Ask Endpoint: Waiting for RAG query future with timeout of {OLLAMA_REQUEST_TIMEOUT_SECONDS} seconds.")
        response = await asyncio.wait_for(
             rag_future,
             timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS
        )
        logging.debug("Ask Endpoint: RAG query future completed within timeout.")

    except asyncio.TimeoutError:
         logging.error(f"Ask Endpoint: RAG query processing timed out after {OLLAMA_REQUEST_TIMEOUT_SECONDS} seconds.")
         # Note: This cancels the Future, but the thread might continue running for a bit
         # depending on where it was blocked. Python threads cannot be forcefully terminated.
         raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=f"RAG query timed out after {OLLAMA_REQUEST_TIMEOUT_SECONDS} seconds. Ollama may be unresponsive or context is too large.")
    except Exception as e:
         logging.error(f"Ask Endpoint: Error during RAG query future execution: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing your question.")


    logging.info("Query processed successfully.")
    return {"answer": response}


# --- Debug Retriever Endpoint (Uses global vector_store with lock and ThreadPoolExecutor) ---
# Update the debug endpoint to also use the ThreadPoolExecutor for consistency and timeout
def perform_retriever_debug(query: str, current_vector_store):
    """
    Synchronous function to perform retriever debug:
    1. Retrieve documents from the vector store.
    This function is designed to be run within a ThreadPoolExecutor.
    """
    logging.debug(f"ThreadPool: Starting retriever debug for: {query}")
    retriever = current_vector_store.as_retriever(search_kwargs={'k': 20}) # Increased k here too for consistency
    retrieved_docs = retriever.invoke(query)
    logging.debug(f"ThreadPool: Retriever debug completed.")
    return retrieved_docs


@app.post("/debug-retriever")
async def debug_retriever(request: QueryRequest):
    """
    Debug endpoint to retrieve documents from the vector store for a query.
    Returns the page_content and metadata of the retrieved documents.
    Runs the retrieval operation in a ThreadPoolExecutor with a timeout.
    """
    # Acquire lock before accessing global vector_store
    with reload_lock:
        logging.debug("Debug Retriever: Acquired reload lock.")
        if vector_store is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector store is not loaded. Index may not be ready or failed to load onto GPU.")

        current_vector_store = vector_store
        logging.debug("Debug Retriever: Released reload lock.")


    logging.info(f"Debugging retriever for query: {request.query}")

    # Submit the synchronous retriever debug function to the thread pool executor
    loop = asyncio.get_running_loop()
    retriever_future = loop.run_in_executor(
        executor,
        perform_retriever_debug,
        request.query,
        current_vector_store
    )

    try:
        # Wait for the future to complete with a timeout
        logging.debug(f"Debug Retriever: Waiting for retriever future with timeout of {OLLAMA_REQUEST_TIMEOUT_SECONDS} seconds.")
        retrieved_docs = await asyncio.wait_for(
            retriever_future,
            timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS
        )
        logging.debug("Debug Retriever: Retriever future completed within timeout.")

    except asyncio.TimeoutError:
         logging.error(f"Debug Retriever: Retrieval operation timed out after {OLLAMA_REQUEST_TIMEOUT_SECONDS} seconds.")
         raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=f"Retrieval operation timed out after {OLLAMA_REQUEST_TIMEOUT_SECONDS} seconds. Ollama embedding may be unresponsive.")
    except Exception as e:
         logging.error(f"Debug Retriever: Error during retriever future execution: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving documents from vector store.")


    # Prepare response data
    response_docs = []
    for doc in retrieved_docs:
         response_docs.append({
             "page_content": doc.page_content,
             "metadata": doc.metadata
         })

    logging.info(f"Debug Retriever: Retrieved {len(response_docs)} documents for debug.")
    return {"query": request.query, "retrieved_documents": response_docs}


# To run the API:
if __name__ == "__main__":
    # Note: --reload might interfere with background tasks.
    # For production, you would typically run without --reload.
    # Use a higher log level for uvicorn itself if you want less verbosity
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")
