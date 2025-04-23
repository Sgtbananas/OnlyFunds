# decorators.py

import time
import functools
import logging

logger = logging.getLogger(__name__)

def log_execution_time(func):
    """Decorator to log the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Execution of {func.__name__} took {execution_time:.4f} seconds")
        return result
    return wrapper

def log_api_call(func):
    """Decorator to log API calls made within the function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling API: {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"API call {func.__name__} completed")
        return result
    return wrapper

def retry_on_failure(retries=3, delay=5):
    """Decorator to retry a function on failure (e.g., API call failures)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    logger.error(f"Error in {func.__name__} (attempt {attempt}/{retries}): {e}")
                    if attempt < retries:
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {retries} attempts failed for {func.__name__}")
                        raise e
        return wrapper
    return decorator

def cache_result(cache, max_cache_size=100):
    """Decorator to cache function results (for expensive calls)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in cache:
                logger.info(f"Cache hit for {func.__name__} with args: {args}, kwargs: {kwargs}")
                return cache[key]
            else:
                result = func(*args, **kwargs)
                if len(cache) >= max_cache_size:
                    cache.clear()  # Clear the cache if it exceeds max size
                cache[key] = result
                logger.info(f"Cache miss for {func.__name__}, result cached.")
                return result
        return wrapper
    return decorator
