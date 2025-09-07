import logging
from src.config import LOG_FILE_PATH
import os

os.makedirs(os.path.dirname(LOG_FILE_PATH),exist_ok=True)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s"
)

logger=logging.getLogger("regression_pipeline")