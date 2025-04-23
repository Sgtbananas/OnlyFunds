# utils/config_loader.py

import os
from dotenv import load_dotenv

load_dotenv()

def get_env_variable(name: str, default=None, cast_type=str):
    val = os.getenv(name, default)
    if cast_type == bool:
        return str(val).lower() in ["1", "true", "yes"]
    if cast_type == list:
        return val.split(",") if val else []
    return cast_type(val)
