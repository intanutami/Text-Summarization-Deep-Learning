from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Path model
model_path = r"C:\Users\vevav\OneDrive\Desktop\coolyeah\07. Sem 7\dl\TextSummarization_API2\model\new_indoT5"

# Load tokenizer dan model
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def summarize_text(
    text: str,
    max_length: int = 150, 
    min_length: int = 30, 
    no_repeat_ngram_size: int = 2,
    num_beams: int = 4,
    # repetition_penalty: float = 3.0,
    # length_penalty: float = 1.2,
    # early_stopping: bool = True,
    # do_sample: bool = True,
    # temperature: float = 1.0,
    # top_k: int = 50,
    # top_p: float = 0.9
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
        # repetition_penalty=repetition_penalty,
        # length_penalty=length_penalty,
        # early_stopping=early_stopping,
        # do_sample=do_sample,
        # temperature=temperature,
        # top_k=top_k,
        # top_p=top_p
    )
    
    # Dekode output menjadi teks
    summary = tokenizer.decode(
        outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return summary
