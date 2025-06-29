from pydantic import BaseModel


class BenchmarkConfig(BaseModel):
    model: str
    continous_batching: bool
    batch_size: int
    max_prefix_len: int
    max_new_tokens: int
    dataset_size: int
