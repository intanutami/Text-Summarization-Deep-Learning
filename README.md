# Text-Summarization-Deep-Learning

Link Model Text Summarization :

Model T5-base
https://huggingface.co/intanutami/clean-model-t5-10epoch-lower

Model mt5-small
https://huggingface.co/intanutami/clean-model-mt5-small-10epoch

Model indo-t5-base
https://huggingface.co/intanutami/clean-model-indo-t5-10epoch-lower

###
###
# Text Summarization API

Currently, the API utilizes **IndoT5-base**, a model optimized for generating accurate text summaries in Indonesian.

## Installation

#### 1. Create a Virtual Environment  
To ensure a clean environment, create a virtual environment using the following command:  
```bash  
python -m venv <your_virtual_env_name>  
```
To start the virtual environment, use this following command
```bash
venv\Scripts\activate
```
###
#### 2. Install depedencies
Install all required dependencies:
```bash
pip install -r /path/to/requirements.txt
```

## Running the API
To start the API, execute the following command in your terminal:
```bash
uvicorn app.main:app --reload
```
Once the server is running, you can access the interactive API documentation via this link: [Text Summarization API](http://127.0.0.1:8000/docs#/)
