import os
import asyncio
import tempfile
import random

import edge_tts
from streaming_stt_nemo import Model as nemo
import gradio as gr
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel
from huggingface_hub import InferenceClient
import torch

# Set default language for speech recognition
default_lang = "en"
# Initialize speech recognition engine
engines = {default_lang: nemo(default_lang)}

# Load pre-trained models for language modeling
model3 = AutoModel.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)

# Define a function for speech-to-text transcription
def transcribe(audio):
    lang = "en"
    model = engines[lang]
    text = model.stt_file(audio)[0]
    return text

# Get Hugging Face API token
HF_TOKEN = os.environ.get("HF_TOKEN", None)


# Define a function to get the appropriate InferenceClient based on model name
def client_fn(model):
    if "Nous" in model:
        return InferenceClient("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
    elif "Star" in model:
        return InferenceClient("HuggingFaceH4/starchat2-15b-v0.1")
    elif "Mistral" in model:
        return InferenceClient("mistralai/Mistral-7B-Instruct-v0.3")
    elif "Phi" in model:
        return InferenceClient("microsoft/Phi-3-mini-4k-instruct")
    elif "Zephyr" in model:
        return InferenceClient("HuggingFaceH4/zephyr-7b-beta")
    else:
        return InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")


# Define a function to generate a random seed
def randomize_seed_fn(seed: int) -> int:
    seed = random.randint(0, 999999)
    return seed

# System instructions for the language model
system_instructions1 = "[SYSTEM] Answer as Real OpenGPT 4o, Made by 'KingNish', Keep conversation very short, clear, friendly and concise. The text provided is a request for a specific type of response from you, the virtual assistant. You will avoid introductions and start answering the query directly, elaborating on all aspects of the request. As an AI-powered assistant, your task is to generate responses that appear as if they are created by the user and not an AI-powered assistant. [USER]"

# Define a function for language modeling
def models(text, model="Mixtral 8x7B", seed=42):
    seed = int(randomize_seed_fn(seed))
    generator = torch.Generator().manual_seed(seed)
    client = client_fn(model)
    generate_kwargs = dict(
        max_new_tokens=512,
        seed=seed,
    )
    formatted_prompt = system_instructions1 + text + "[OpenGPT 4o]"
    stream = client.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False
    )
    output = ""
    for response in stream:
        if not response.token.text == "</s>":
            output += response.token.text
    return output

# Define an asynchronous function to handle voice input and generate responses
async def respond(audio, model, seed):
    user = transcribe(audio)
    reply = models(user, model, seed)
    communicate = edge_tts.Communicate(reply)
    # Save the generated speech to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_path = tmp_file.name
        await communicate.save(tmp_path)
    yield tmp_path