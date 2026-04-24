'''
YAML-based configuration
Environment variable overrides
Validation with Pydantic
Support for dev/staging/prod environments
'''
import os
import yaml
from pydantic import BaseModel, Field, ValidationError


class Config(BaseModel):
    chunk_size: int = Field(1000, description="Size of text chunks")
    overlap: int = Field(200, description="Overlap between chunks")
    data_dir: str = Field("data", description="Directory for input files")
    output_dir: str = Field("output", description="Directory for processed output")
    log_level: str = Field("INFO", description="Logging level")


def load_config(config_path: str = "config.yaml", env: str | None = None) -> Config:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f) or {}

    # Merge environment-specific overrides (e.g. config.dev.yaml)
    resolved_env = env or os.getenv("ENV", "dev")
    base, ext = os.path.splitext(config_path)
    env_config_path = f"{base}.{resolved_env}{ext}"
    if os.path.exists(env_config_path):
        with open(env_config_path, 'r') as f:
            env_overrides = yaml.safe_load(f) or {}
        config_data.update(env_overrides)

    # Override with environment variables (e.g. CHUNK_SIZE=500)
    for field_name, field_info in Config.model_fields.items():
        env_value = os.getenv(field_name.upper())
        if env_value is not None:
            annotation = field_info.annotation
            config_data[field_name] = annotation(env_value)

    try:
        return Config(**config_data)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")
