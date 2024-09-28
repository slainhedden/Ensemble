import logging
from functools import wraps

logging.getLogger("openai").disabled = True
logging.getLogger("httpx").disabled = True

def error_handler(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            # You might want to implement some recovery logic here
    return wrapper

# Use this decorator on your async functions
# @error_handler
# async def some_function():
#     ...