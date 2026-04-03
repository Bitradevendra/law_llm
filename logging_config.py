import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(log_dir='logs', log_file='train.log', level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger 