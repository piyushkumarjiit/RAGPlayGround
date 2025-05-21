# ollama_monitor_service.py
# This script monitors the health of the Ollama server by periodically
# checking if an available LLM model is responsive via a small generation request.
# If Ollama is unresponsive for a configurable number of consecutive checks,
# it attempts to restart the Ollama system service (designed for systemd on Linux).

import time
import requests
import subprocess
import logging
import sys
import os
import threading
from datetime import datetime
import json # Import json for parsing API responses

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434" # Default Ollama API URL
CHECK_INTERVAL_SECONDS = 30 # How often to check Ollama's health
TIMEOUT_SECONDS_PER_CHECK = 20 # Timeout for each individual API request to Ollama (tags or generate)
MAX_CONSECUTIVE_FAILURES = 5 # How many consecutive failures before attempting restart
OLLAMA_SERVICE_NAME = "ollama" # The name of the Ollama systemd service
# LLM_MODEL_TO_CHECK is now a fallback/preferred model, not strictly hardcoded
# Set to None or an empty string to always pick the first available model
# Set to a specific model name (e.g., "qwen3:8b") to prefer that model if available
PREFERRED_LLM_MODEL_TO_CHECK = "qwen3:8b" # Preferred LLM model for health check

# --- Global Variables ---
consecutive_failures = 0
is_restarting = False # Flag to prevent multiple restart attempts simultaneously
monitor_thread = None # To hold the monitoring thread


# --- Logging Setup ---
# Configure logging to a file and the console
logging.basicConfig(
    level=logging.INFO, # Set minimum level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ollama_monitor_service.log', # Specify the log file name
    filemode='a' # Append to the log file if it exists
)

# Add a console handler for visibility
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO) # Set level for console output
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)
# -------------------------

def get_available_ollama_models():
    """Queries Ollama's /api/tags endpoint to get a list of available models."""
    logging.debug(f"Querying Ollama for available models at {OLLAMA_API_URL}/api/tags")
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=TIMEOUT_SECONDS_PER_CHECK)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        # Extract model names - the 'models' list contains dicts with a 'name' key
        models = [model['name'] for model in data.get('models', [])]
        logging.debug(f"Found {len(models)} available models: {models}")
        return models
    except requests.exceptions.Timeout:
        logging.warning(f"Timeout while querying /api/tags after {TIMEOUT_SECONDS_PER_CHECK} seconds.")
        return None # Indicate failure to get models
    except requests.exceptions.ConnectionError:
        logging.warning(f"Connection error while querying /api/tags. Is Ollama running at {OLLAMA_API_URL}?")
        return None # Indicate failure to get models
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while querying /api/tags: {e}")
        return None # Indicate failure to get models
    except json.JSONDecodeError:
        logging.error("Received non-JSON response from /api/tags.")
        return None # Indicate failure to get models
    except Exception as e:
        logging.error(f"An unexpected error occurred while getting available models: {e}", exc_info=True)
        return None # Indicate failure to get models


def check_ollama_health():
    """
    Checks Ollama health by:
    1. Getting available models from /api/tags.
    2. Selecting a model (preferred or first available).
    3. Sending a small LLM generation request using the selected model.
    """
    logging.debug("Starting Ollama health check sequence.")

    # Step 1: Get available models
    available_models = get_available_ollama_models()

    if available_models is None:
        logging.warning("Failed to retrieve available models from Ollama.")
        return False # Health check fails if we can't even get the model list

    if not available_models:
        logging.warning("No LLM models found in Ollama.")
        logging.warning("Please pull a model using: `ollama pull <model_name>`")
        return False # Health check fails if no models are available


    # Step 2: Select a model for the health check
    model_to_use = None
    if PREFERRED_LLM_MODEL_TO_CHECK and PREFERRED_LLM_MODEL_TO_CHECK in available_models:
        model_to_use = PREFERRED_LLM_MODEL_TO_CHECK
        logging.debug(f"Using preferred health check model: '{model_to_use}'")
    else:
        # If preferred model is not set or not available, use the first available model
        model_to_use = available_models[0]
        if PREFERRED_LLM_MODEL_TO_CHECK: # Only warn if a preferred model was specified but not found
             logging.warning(f"Preferred health check model '{PREFERRED_LLM_MODEL_TO_CHECK}' not found. Using first available model: '{model_to_use}'")
        else:
             logging.debug(f"No preferred model specified. Using first available model: '{model_to_use}'")


    # Step 3: Perform a small LLM generation request using the selected model
    logging.debug(f"Performing LLM generation health check using model: {model_to_use}")
    try:
        payload = {
            "model": model_to_use,
            "prompt": "Hi", # A very simple prompt
            "stream": False, # Don't stream, wait for full response
            "options": {
                "num_predict": 5, # Generate only a few tokens
                "temperature": 0,
            }
        }
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json=payload,
            timeout=TIMEOUT_SECONDS_PER_CHECK # Use the defined timeout
        )
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Basic check of the response structure
        try:
            data = response.json()
            if "response" in data and isinstance(data["response"], str):
                 logging.debug("Ollama health check (LLM generation) successful.")
                 return True
            else:
                 logging.warning("Ollama health check (LLM generation) received unexpected response format.")
                 return False
        except json.JSONDecodeError:
             logging.warning("Ollama health check (LLM generation) received non-JSON response.")
             return False


    except requests.exceptions.Timeout:
        logging.warning(f"Ollama health check (LLM generation) timed out after {TIMEOUT_SECONDS_PER_CHECK} seconds.")
        return False
    except requests.exceptions.ConnectionError:
        logging.warning(f"Ollama connection error during LLM generation health check. Is Ollama running at {OLLAMA_API_URL}?")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred during Ollama LLM generation health check: {e}")
        # Specific check for model not found is now handled by the /api/tags check
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during Ollama health check: {e}", exc_info=True)
        return False


def restart_ollama_service():
    """Attempts to restart the Ollama systemd service."""
    global is_restarting
    if is_restarting:
        logging.info("Restart already in progress. Skipping new restart request.")
        return

    is_restarting = True
    logging.warning(f"\n--- Attempting to restart Ollama service ({OLLAMA_SERVICE_NAME}) ---")

    try:
        # Check if systemctl command exists (assuming systemd)
        try:
            subprocess.run(["which", "systemctl"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.error("❌ systemctl command not found. Cannot restart Ollama service.")
            logging.error("This script assumes Ollama is running as a systemd service on Linux.")
            logging.error("Manual intervention is required to restart Ollama.")
            is_restarting = False
            return


        # Use subprocess.run with check=True to automatically raise CalledProcessError on failure
        # systemctl requires root privileges, so this script should be run with sudo
        logging.info(f"Stopping service: sudo systemctl stop {OLLAMA_SERVICE_NAME}")
        subprocess.run(["systemctl", "stop", OLLAMA_SERVICE_NAME], check=True, capture_output=True, text=True)
        logging.info("✅ Ollama service stopped.")

        # Add a small delay before starting
        time.sleep(5)

        logging.info(f"Starting service: sudo systemctl start {OLLAMA_SERVICE_NAME}")
        subprocess.run(["systemctl", "start", OLLAMA_SERVICE_NAME], check=True, capture_output=True, text=True)
        logging.info("✅ Ollama service started.")

        # Add a delay for Ollama to initialize and load models
        time.sleep(30) # Increased delay after restart for models to load

        logging.info("--- Ollama service restart sequence complete. ---")
        # Reset failure count after a successful restart attempt (even if it fails later)
        global consecutive_failures
        consecutive_failures = 0

    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to restart Ollama service using systemctl: {e.cmd}")
        logging.error(f"Return Code: {e.returncode}")
        logging.error(f"Stderr: {e.stderr}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error("Please check systemctl logs for the Ollama service for more details:")
        logging.error(f"  journalctl -u {OLLAMA_SERVICE_NAME}")
        logging.error("Manual intervention may be required.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during Ollama service restart: {e}", exc_info=True)
        logging.error("Manual intervention may be required.")
    finally:
        is_restarting = False


def monitor_loop():
    """Main monitoring loop."""
    global consecutive_failures

    logging.info(f"Starting Ollama monitor loop. Checking every {CHECK_INTERVAL_SECONDS} seconds.")
    logging.info(f"Restart will be attempted after {MAX_CONSECUTIVE_FAILURES} consecutive failures.")
    # The specific model used for the check is now determined dynamically,
    # but we can log the *preferred* model if set.
    if PREFERRED_LLM_MODEL_TO_CHECK:
         logging.info(f"Configured preferred health check model: '{PREFERRED_LLM_MODEL_TO_CHECK}'")
    else:
         logging.info("No preferred health check model configured. Will use the first available model.")


    # Initial delay before first check
    time.sleep(5)

    while True:
        # Use datetime.now() directly since datetime class is imported
        logging.debug(f"Performing Ollama health check at {datetime.now()}")
        if check_ollama_health():
            if consecutive_failures > 0: # Only log reset if there were previous failures
                logging.info(f"Ollama is healthy. Resetting consecutive failures from {consecutive_failures} to 0.") # Log the reset
            else:
                logging.debug("Ollama is healthy.") # Keep debug for normal healthy checks
            consecutive_failures = 0 # Reset failure count on success
        else:
            consecutive_failures += 1
            logging.warning(f"Ollama health check failed. Consecutive failures: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}")

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logging.error(f"Reached maximum consecutive failures ({MAX_CONSECUTIVE_FAILURES}). Attempting to restart Ollama service.")
                restart_ollama_service()
                # After attempting restart, wait a bit before the next check
                time.sleep(30) # Wait longer after a restart attempt
                continue # Skip the rest of the current loop iteration


        # Wait for the next check
        time.sleep(CHECK_INTERVAL_SECONDS)


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Initializing Ollama Monitor Service ---")

    # Check if running with root privileges (required for systemctl)
    if os.geteuid() != 0:
        logging.critical("FATAL: This script requires root privileges to restart the Ollama system service.")
        logging.critical("Please run it using: `sudo python ollama_monitor_service.py`")
        sys.exit(1)
    logging.info("Running with root privileges.")
    logging.info(f"Monitoring Ollama at: {OLLAMA_API_URL}")


    try:
        # Start the monitoring loop in a separate thread
        # This allows the main thread to potentially handle signals or other tasks if needed
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True) # daemon=True allows script to exit if main thread exits
        monitor_thread.start()
        logging.info("Ollama monitor thread started.")
        logging.info("Monitor Service is running. Press Ctrl+C to stop.")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("\nCtrl+C detected. Stopping Ollama Monitor Service.")
    except Exception as e:
         logging.critical(f"An unexpected fatal error occurred during Monitor Service execution: {e}", exc_info=True)
    finally:
        # The daemon thread will exit when the main thread exits
        logging.info("Ollama Monitor Service finished.")
