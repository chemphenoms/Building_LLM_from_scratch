import logging


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up a Python logger with a specified name and logging level.

    Args:
        name (str): Unique name for the logger (use module name typically).
        level (int): Logging level (default: logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """

    # Create or get a named logger
    logger = logging.getLogger(name)

    # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger already configured
    if not logger.handlers:
        # Create a handler for console output
        console_handler = logging.StreamHandler()

        # Define the format for the log messages
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Attach the formatter to the console handler
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

        # Optional: avoid logging messages being propagated to root logger
        logger.propagate = False

    return logger
