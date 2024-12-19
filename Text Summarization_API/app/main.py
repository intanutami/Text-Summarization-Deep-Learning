from fastapi import FastAPI, HTTPException
from app.model import summarize_text
from app.schemas import SummarizationRequest, SummarizationResponse

# Inisialisasi FastAPI
app = FastAPI(title="Text Summarization API")

@app.get("/")
def home():
    return {"message": "Welcome to the Text Summarization API!"}

@app.post("/summarize", response_model=SummarizationResponse)
def summarize(request: SummarizationRequest):
    """
    Endpoint untuk merangkum teks.
    """
    if len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    text_lowercase = request.text.lower()

    # Panggil fungsi summarize_text dengan semua parameter
    summary = summarize_text(
        text=text_lowercase,
        max_length=request.max_length,
        min_length=request.min_length,
        no_repeat_ngram_size=request.no_repeat_ngram_size,
        num_beams=request.num_beams,
        # repetition_penalty=request.repetition_penalty,
        # length_penalty=request.length_penalty,
        # early_stopping=request.early_stopping,
        # do_sample=request.do_sample,
        # temperature=request.temperature,
        # top_k=request.top_k,
        # top_p=request.top_p,
    )
    
    return SummarizationResponse(summary=summary)
