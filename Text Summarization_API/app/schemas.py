from pydantic import BaseModel

class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 150  # Sama dengan training
    min_length: int = 30  # Sama dengan training
    no_repeat_ngram_size: int = 2
    num_beams: int = 4
    # repetition_penalty: float = 3.0
    # length_penalty: float = 1.2
    # early_stopping: bool = True
    # do_sample: bool = True
    # temperature: float = 1.0
    # top_k: int = 50
    # top_p: float = 0.9

class SummarizationResponse(BaseModel):
    summary: str