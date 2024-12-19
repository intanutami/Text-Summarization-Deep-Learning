from pydantic import BaseModel

class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30 
    no_repeat_ngram_size: int = 2
    num_beams: int = 4

class SummarizationResponse(BaseModel):
    summary: str
