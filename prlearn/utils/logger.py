import logging
import os


def get_logger(name: str = __name__) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s:%(module)s:%(funcName)s:%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    return logger
