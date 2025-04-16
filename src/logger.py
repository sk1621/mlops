import logging
import os
from datetime import datetime

# Generate file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define folder path
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)  # Create only the folder

# Final path = logs_dir + LOG_FILE
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)



# Setup logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging started well.")
    print("Logger ran!") 
