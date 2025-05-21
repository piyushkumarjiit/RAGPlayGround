import os
import json
import time
import requests
import logging
import re # Import the re module for regular expressions
import argparse # Import argparse for command-line arguments

# Import necessary loaders and classes from langchain
# You will need to install these:
# pip install langchain langchain-community pypdf docx2txt unstructured openpyxl python-pptx
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter # Ensure this is imported
from langchain.docstore.document import Document # Ensure this is imported

# --- Logging Configuration ---
# Set level to DEBUG to see all detailed messages, including raw LLM output
# Set to INFO for less verbose output once debugging is complete
# Configure logging to a file and console
log_file_path = "qna_generation.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'), # Log to file, overwrite each run
        logging.StreamHandler() # Log to console
    ]
)
logging.getLogger('httpx').setLevel(logging.WARNING) # Suppress noisy http client logs
logging.getLogger('httpcore').setLevel(logging.WARNING) # Suppress noisy http client logs


# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
QWEN_MODEL_NAME = "qwen3:8b" # Updated to your specific model
DOCUMENTS_DIR = "data" # Directory where your source documents are located
OUTPUT_DIR = "generated_qna_data" # Directory to save generated Q&A JSON files
PAGES_TO_SKIP_AT_START = 10 # Number of initial pages to skip for PDF documents
PAGES_TO_SKIP_AT_END = 0 # Number of pages to skip from the end for PDF documents

# --- Chunking Parameters (tuned for Q&A generation) ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
QUESTIONS_PER_CHUNK = 10 # Aim for exactly 10 questions per chunk

# --- Create output directory if it doesn't exist ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Text Cleaning Function ---
def clean_text(text: str) -> str:
    """
    Cleans the input text by:
    1. Removing common non-printable characters (like form feed, carriage returns, etc.).
    2. Replacing multiple spaces, tabs, and newlines with a single space or newline.
    3. Stripping leading/trailing whitespace from each line and the whole text.
    4. Removing excessive empty lines (more than two consecutive newlines).
    """
    # Remove common non-printable characters (e.g., form feed \x0c, vertical tab \x0b)
    # and control characters that might be artifacts of PDF extraction.
    # Keep standard whitespace characters (\s, \n, \r, \t) for now to normalize them later.
    cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Replace multiple spaces/tabs with a single space
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
    
    # Replace multiple newlines with at most two newlines (for paragraph separation)
    # This also effectively handles page breaks if they were just extra newlines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text) 
    
    # Replace carriage return with newline and normalize consecutive newlines again
    cleaned_text = cleaned_text.replace('\r', '\n')
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    # Strip whitespace from each line and then from the entire string
    cleaned_lines = [line.strip() for line in cleaned_text.split('\n')]
    cleaned_text = '\n'.join(cleaned_lines).strip()

    return cleaned_text


# --- Document Loading (adapted from your indexer_service.py) ---
def load_document_with_path(file_path: str, skip_start_pages: int, skip_end_pages: int) -> list:
    """
    Loads a single document, applies page skipping from start and end (for PDFs),
    concatenates its page content (if multi-page), cleans the text, and returns it
    as a single Langchain Document object.
    This effectively removes internal 'page breaks' by merging all content.
    """
    logging.info(f"Attempting to load file: {file_path}")
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        raw_docs = []

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
            all_pages = loader.load()
            total_pages = len(all_pages)
            
            logging.info(f"PDF document {file_path} has {total_pages} pages.")

            # Apply page skipping from the start
            if skip_start_pages > 0 and total_pages > skip_start_pages:
                raw_docs = all_pages[skip_start_pages:]
                logging.info(f"Skipped first {skip_start_pages} pages for PDF: {file_path}. Remaining pages: {len(raw_docs)}")
            else:
                raw_docs = all_pages
                if skip_start_pages > 0: # Only log if user tried to skip
                    logging.info(f"Document {file_path} has {total_pages} pages, less than or equal to {skip_start_pages} pages. No pages skipped from start.")
            
            # Apply page skipping from the end to the *remaining* pages
            if skip_end_pages > 0 and len(raw_docs) > skip_end_pages:
                raw_docs = raw_docs[:-skip_end_pages]
                logging.info(f"Skipped last {skip_end_pages} pages for PDF: {file_path}. Remaining pages: {len(raw_docs)}")
            elif skip_end_pages > 0: # Only log if user tried to skip
                logging.info(f"After skipping start pages, document {file_path} has {len(raw_docs)} pages, less than or equal to {skip_end_pages} pages. No pages skipped from end.")
            
        elif file_extension in (".md", ".txt"):
            loader = TextLoader(file_path)
            raw_docs = loader.load()
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
            raw_docs = loader.load()
        elif file_extension == ".csv":
            loader = CSVLoader(file_path)
            raw_docs = loader.load()
        elif file_extension in (".xlsx", ".xls"):
            logging.debug(f"Using UnstructuredExcelLoader for file: {file_path}")
            loader = UnstructuredExcelLoader(file_path)
            raw_docs = loader.load()
        elif file_extension in (".pptx", ".ppt"):
            logging.debug(f"Using UnstructuredPowerPointLoader for file: {file_path}")
            loader = UnstructuredPowerPointLoader(file_path)
            raw_docs = loader.load()
        else:
            logging.info(f"Skipping unsupported file format: {file_path}")
            return None # Return None for unsupported types

        if not raw_docs:
            logging.warning(f"No raw content loaded for {file_path}.")
            return None

        # Concatenate all page contents into a single string
        full_content = "\n\n".join([doc.page_content for doc in raw_docs])
        
        # Clean the concatenated content
        cleaned_full_content = clean_text(full_content)

        # Create a single Document object for the entire file
        # We try to preserve metadata from the first raw_doc, and add source if missing
        file_metadata = raw_docs[0].metadata.copy() if raw_docs else {}
        if 'source' not in file_metadata:
            file_metadata['source'] = os.path.abspath(file_path)

        single_doc = Document(page_content=cleaned_full_content, metadata=file_metadata)
        
        logging.info(f"Loaded and consolidated content from {len(raw_docs)} original parts into a single document for {file_path} (cleaned length: {len(cleaned_full_content)} chars).")
        return [single_doc] # Return as a list containing one Document

    except Exception as e: # This try/except is for the load_document_with_path function itself
        logging.error(f"Error loading file {file_path}: {e}", exc_info=True)
        return None


def get_current_files_state(paths_to_scan: list[str]) -> dict[str, float]:
    """Scans the monitored paths and returns the current state of files (path: mtime)."""
    current_state = {}
    logging.info(f"Scanning paths for current file state: {paths_to_scan}")
    if not paths_to_scan:
        logging.warning("No paths specified to scan.")
        return {}

    supported_extensions = (".pdf", ".md", ".txt", ".docx", ".csv", ".xlsx", ".xls", ".pptx", ".ppt")

    for data_path in paths_to_scan:
        if not os.path.exists(data_path):
            logging.warning(f"Scan path does not exist: {data_path}. Skipping.")
            continue

        for root, _, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                if not os.path.isfile(file_path) or file.startswith('.'):
                    continue
                if not file_path.lower().endswith(supported_extensions):
                    continue

                try:
                    mtime = os.path.getmtime(os.path.abspath(file_path))
                    current_state[os.path.abspath(file_path)] = mtime
                except Exception as e:
                    logging.error(f"Error getting mtime for file {file_path}: {e}", exc_info=True)

    logging.info(f"Scanned {len(current_state)} files.")
    return current_state

# --- Initialize Text Splitter for Q&A Generation ---
text_splitter_for_qna = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False, # Set to True if your separators are regex patterns
    separators=[
        "\n\n\n",  # Try to split by triple newline (potential section breaks)
        "\n\n",   # Then by double newline (paragraphs)
        "\n",     # Then by single newline (lines)
        " ",      # Then by spaces (words)
        "",       # Finally, by characters
    ]
)

# --- LLM Prompt Template ---
LLM_PROMPT_TEMPLATE = """
You are an expert content analyzer for technical user manuals. Your goal is to generate a list of question-answer pairs strictly from the provided text snippet.

For each question, the answer MUST be directly and completely derivable from the provided "Text Snippet". DO NOT use any external knowledge.
Each question should be unique and relevant to a user trying to understand the manual.
Generate exactly {num_questions} distinct question-answer pairs if possible. If the snippet is too short, generate fewer but ensure they are all grounded.

Format your output as a JSON array of objects. Each object must have three keys:
- "question": (string) The question.
- "answer": (string) The exact answer from the "Text Snippet".
- "source_text_snippet": (string) The original text snippet provided, which contains the answer.

Text Snippet:
\"\"\"
{text_chunk}
\"\"\"

JSON Output:
"""

# --- Main Generation Logic ---
def generate_qna_for_document(doc_path: str, overwrite_flag: bool = False, skip_start: int = 0, skip_end: int = 0) -> dict:
    """
    Generates Q&A pairs for a single document.
    Returns a dictionary with generation status (processed, skipped, failed, qna_count).
    """
    logging.info(f"Processing document: {doc_path}")
    
    # --- MODIFIED: Generate unique filename using os.path.splitext ---
    # This will take the full filename before the last dot (extension), ensuring more uniqueness
    doc_file_name_base = os.path.splitext(os.path.basename(doc_path))[0]
    # --- END MODIFIED ---

    output_file_path = os.path.join(OUTPUT_DIR, f"{doc_file_name_base}_generated_qna.json")

    # Check if Q&A file already exists and overwrite_flag is False
    if os.path.exists(output_file_path) and not overwrite_flag:
        logging.info(f"Q&A file already exists for {doc_file_name_base} at {output_file_path}. Skipping generation. Use --overwrite to regenerate.")
        return {"status": "skipped", "reason": "exists", "qna_count": 0}

    # Pass skip_start_pages and skip_end_pages to load_document_with_path
    consolidated_docs = load_document_with_path(doc_path, skip_start_pages=skip_start, skip_end_pages=skip_end)
    if not consolidated_docs: 
        logging.warning(f"No content to process for {doc_path} after skipping pages. Skipping Q&A generation.")
        return {"status": "skipped", "reason": "no_content_after_skip", "qna_count": 0}

    # The text splitter will now operate on this single, long document
    chunks = text_splitter_for_qna.split_documents(consolidated_docs)
    
    all_qna_data = []

    for i, chunk_doc in enumerate(chunks):
        chunk_text = chunk_doc.page_content # This content is already cleaned and potentially concatenated
        
        logging.info(f"  Processing chunk {i+1}/{len(chunks)} for {doc_file_name_base} (length: {len(chunk_text)} chars)...")
        prompt = LLM_PROMPT_TEMPLATE.format(num_questions=QUESTIONS_PER_CHUNK, text_chunk=chunk_text)

        payload = {
            "model": QWEN_MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            
            llm_output = response.json().get("response", "")
            # --- Trim leading/trailing whitespace from the raw LLM output ---
            llm_output = llm_output.strip()
            # --- END Trim ---
            
            # --- DEBUGGING: Print raw LLM output (now trimmed) ---
            logging.debug(f"    Raw LLM output for chunk {i+1} (trimmed):\n{llm_output}")
            # --- END DEBUGGING ---

            # --- IMPROVED JSON PARSING AND VALIDATION ---
            json_string_to_parse = llm_output
            generated_qna = [] # Initialize as empty list in case of parsing failures
            
            try:
                # Try to parse the whole output as JSON directly
                parsed_json_output = json.loads(json_string_to_parse)

                # --- NEW LOGIC TO HANDLE SINGLE OBJECT VS. ARRAY ---
                if isinstance(parsed_json_output, dict):
                    # If it's a single dictionary, wrap it in a list
                    generated_qna = [parsed_json_output]
                    logging.info(f"    LLM returned a single JSON object for chunk {i+1}. Wrapped it in a list.")
                elif isinstance(parsed_json_output, list):
                    # If it's already a list, use it directly
                    generated_qna = parsed_json_output
                else:
                    # If it's neither a dict nor a list, something unexpected happened
                    logging.warning(f"    LLM output for chunk {i+1} was not a JSON array or object as expected. Type: {type(parsed_json_output)}. Skipping Q&A generation for this chunk.")
                    logging.debug(f"    Parsed content: {parsed_json_output}")
                    generated_qna = [] # Ensure it's empty to skip further processing for this chunk
                    
            except json.JSONDecodeError:
                # If direct parsing fails, try to find the outermost JSON structure
                json_start = -1
                json_end = -1
                
                if '[' in llm_output and ']' in llm_output:
                    json_start = llm_output.find('[')
                    json_end = llm_output.rfind(']') + 1
                elif '{' in llm_output and '}' in llm_output:
                    json_start = llm_output.find('{')
                    json_end = llm_output.rfind('}') + 1

                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_string_to_parse = llm_output[json_start:json_end]
                    # --- Trim leading/trailing whitespace from the extracted JSON string ---
                    json_string_to_parse = json_string_to_parse.strip()
                    # --- END Trim ---
                    try:
                        parsed_json_output = json.loads(json_string_to_parse)
                        # Apply the same wrapping logic for extracted JSON
                        if isinstance(parsed_json_output, dict):
                            generated_qna = [parsed_json_output]
                            logging.info(f"    LLM returned a single JSON object (extracted) for chunk {i+1}. Wrapped it in a list.")
                        elif isinstance(parsed_json_output, list):
                            generated_qna = parsed_json_output
                        else:
                            logging.warning(f"    Extracted LLM output for chunk {i+1} was not a JSON array or object. Type: {type(parsed_json_output)}. Skipping.")
                            logging.debug(f"    Parsed content (extracted): {parsed_json_output}")
                            generated_qna = []
                    except json.JSONDecodeError as inner_json_err:
                        logging.error(f"    Inner JSON decoding error for chunk {i+1} after extraction: {inner_json_err}")
                        logging.error(f"    JSON string attempted to parse: {json_string_to_parse}")
                        generated_qna = []
                else:
                    logging.warning(f"    Could not find a clear JSON array or object in LLM output for chunk {i+1}. Skipping Q&A generation for this chunk.")
                    logging.debug(f"    Raw LLM output (no JSON structure found): {llm_output}")
                    generated_qna = []
            # --- END IMPROVED JSON PARSING AND VALIDATION ---
            
            # This check is now less critical for type error, but still good for empty lists
            if not generated_qna: # If generated_qna is still empty or not a list after parsing attempts
                 logging.warning(f"    No Q&A data successfully parsed from LLM output for chunk {i+1}.")
                 continue # This continue was already here and is fine.


            for qa_pair in generated_qna:
                # Ensure qa_pair is a dictionary before trying to assign 'id'
                if not isinstance(qa_pair, dict):
                    logging.warning(f"    Item in generated Q&A was not a dictionary for chunk {i+1}. Skipping item: {qa_pair}")
                    continue # Skip this malformed item

                # --- Apply strip to question, answer, and source_text_snippet ---
                if "question" in qa_pair and isinstance(qa_pair["question"], str):
                    qa_pair["question"] = qa_pair["question"].strip()
                if "answer" in qa_pair and isinstance(qa_pair["answer"], str):
                    qa_pair["answer"] = qa_pair["answer"].strip()
                # source_text_snippet is already based on the chunk, which is now cleaned and concatenated,
                # but trimming its value in the final JSON is good for consistency.
                if "source_text_snippet" in qa_pair and isinstance(qa_pair["source_text_snippet"], str):
                    qa_pair["source_text_snippet"] = qa_pair["source_text_snippet"].strip()
                # --- END Apply strip ---

                # Add a unique ID to each Q&A pair
                qa_pair["id"] = f"{doc_file_name_base}_q_{i+1}_{len(all_qna_data)}"
                
                # Ensure 'source_text_snippet' is present, even if LLM missed it or it's partial
                # The chunk_text itself is already cleaned and concatenated
                if "source_text_snippet" not in qa_pair or not qa_pair["source_text_snippet"]:
                    qa_pair["source_text_snippet"] = chunk_text 
                
                all_qna_data.append(qa_pair)
            
            logging.info(f"    Generated {len(generated_qna)} Q&A pairs from this chunk.")

        except requests.exceptions.RequestException as req_err:
            logging.error(f"    Network or API error for chunk {i+1}: {req_err}")
            return {"status": "failed", "reason": f"API Error: {req_err}", "qna_count": 0}
        except Exception as e: # Catch any other unexpected errors
            logging.error(f"    An unexpected error occurred for chunk {i+1}: {e}", exc_info=True)
            return {"status": "failed", "reason": f"Unexpected error: {e}", "qna_count": 0}
        
        time.sleep(0.5) # Add a small delay to avoid overwhelming Ollama

    if all_qna_data: # Only save if we actually generated some Q&A
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_qna_data, f, indent=2)
        logging.info(f"Finished {doc_file_name_base}. Total Q&A pairs: {len(all_qna_data)}. Saved to {output_file_path}\n")
        return {"status": "processed", "reason": "success", "qna_count": len(all_qna_data)}
    else:
        logging.warning(f"No Q&A pairs generated for {doc_file_name_base}. Output file not created.\n")
        return {"status": "failed", "reason": "no_qna_generated", "qna_count": 0}

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from documents using Ollama.")
    parser.add_argument("--overwrite", action="store_true", 
                        help="Set this flag to regenerate Q&A files even if they already exist.")
    parser.add_argument("--skip-start-pages", type=int, default=PAGES_TO_SKIP_AT_START,
                        help=f"Number of pages to skip from the beginning of PDF documents (default: {PAGES_TO_SKIP_AT_START}).")
    parser.add_argument("--skip-end-pages", type=int, default=PAGES_TO_SKIP_AT_END,
                        help=f"Number of pages to skip from the end of PDF documents (default: {PAGES_TO_SKIP_AT_END}).")
    args = parser.parse_args()

    logging.info("Starting Q&A generation script.")
    start_time = time.time()

    monitored_paths = [DOCUMENTS_DIR]
    current_files = get_current_files_state(monitored_paths)
    
    total_files_scanned = len(current_files)
    files_processed_successfully = 0
    files_skipped_exist = 0
    files_skipped_no_content_after_skip = 0
    files_failed = 0
    total_qna_generated = 0

    if not current_files:
        logging.warning(f"No supported documents found in '{DOCUMENTS_DIR}'. Please ensure files are in the directory and supported types.")

    for file_path in current_files.keys():
        result = generate_qna_for_document(
            file_path,
            overwrite_flag=args.overwrite,
            skip_start=args.skip_start_pages,
            skip_end=args.skip_end_pages
        )
        
        if result["status"] == "processed":
            files_processed_successfully += 1
            total_qna_generated += result["qna_count"]
        elif result["status"] == "skipped":
            if result["reason"] == "exists":
                files_skipped_exist += 1
            elif result["reason"] == "no_content_after_skip":
                files_skipped_no_content_after_skip += 1
        elif result["status"] == "failed":
            files_failed += 1
    
    end_time = time.time()
    duration = end_time - start_time

    logging.info("\n--- Q&A Generation Summary ---")
    logging.info(f"Total files scanned: {total_files_scanned}")
    logging.info(f"Files processed successfully: {files_processed_successfully}")
    logging.info(f"Files skipped (already exist): {files_skipped_exist}")
    logging.info(f"Files skipped (no content after page trimming/unsupported format): {files_skipped_no_content_after_skip}")
    logging.info(f"Files failed during processing: {files_failed}")
    logging.info(f"Total Q&A pairs generated: {total_qna_generated}")
    logging.info(f"Total execution time: {duration:.2f} seconds")
    logging.info(f"Detailed logs available in: {log_file_path}")
    logging.info("Q&A generation complete for all processed documents.")
    logging.info(f"Please review the generated JSON files in the '{OUTPUT_DIR}' directory.")
    logging.info("Manual review is crucial to ensure quality and correctness of generated Q&A.")