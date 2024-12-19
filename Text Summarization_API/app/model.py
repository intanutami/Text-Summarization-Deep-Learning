from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model_path = "https://huggingface.co/intanutami/clean-model-indo-t5-10epoch-lower"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def summarize_text(
    text: str,
    max_length: int = 150, 
    min_length: int = 30, 
    no_repeat_ngram_size: int = 2,
    num_beams: int = 4,
) -> str:
    """
    Fungsi untuk merangkum teks menggunakan model T5 dengan parameter yang dapat disesuaikan.
    """

    inputs = tokenizer(
        text,
        max_length=512,  
        padding="max_length",  
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        min_length=min_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
    )
    
    # Decode output menjadi teks
    summary = tokenizer.decode(
        outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return summary
