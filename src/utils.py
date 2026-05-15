import time
import logging
import numpy as np

RANDOM_STATE = 42
# Seed is set once in main() — not at module import level (ISSUE-4)

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(message)s')
logger = logging.getLogger(__name__)


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def log_section(title):
    logger.info(f"\n{'='*60}\n{title}\n{'='*60}")
