import os
import time
import requests
import random
from threading import Thread
from typing import List, Dict, Union
import subprocess
subprocess.run(
    "pip install flash-attn --no-build-isolation",
    env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    shell=True,
)
import torch
import gradio as gr
from bs4 import BeautifulSoup
from transformers import LlavaProcessor, LlavaForConditionalGeneration, TextIteratorStreamer
from huggingface_hub import InferenceClient
from PIL import Image
import spaces
from functools import lru_cache
import cv2
import re
import io 
import json
from gradio_client import Client, file
from groq import Groq

# You can also use models that are commented below
# model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
model_id = "llava-hf/llava-interleave-qwen-7b-hf"
# model_id = "llava-hf/llava-interleave-qwen-7b-dpo-hf"
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id,torch_dtype=torch.float16,  use_flash_attention_2=True, low_cpu_mem_usage=True)
model.to("cuda")
# Credit to merve for code of llava interleave qwen

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)

client_groq = Groq(api_key=GROQ_API_KEY)

def sample_frames(video_file) :
    try:
        video = cv2.VideoCapture(video_file)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = 12
        interval = total_frames // num_frames
        frames = []
        for i in range(total_frames):
            ret, frame = video.read()
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not ret:
                continue
            if i % interval == 0:
                frames.append(pil_img)
        video.release()
        return frames
    except:
        frames=[]
        return frames
        

# Path to example images
examples_path = os.path.dirname(__file__)
EXAMPLES = [
    [
        {
            "text": "What is Friction? Explain in Detail.",
        }
    ],
    [
        {
            "text": "Write me a Python function to generate unique passwords.",
        }
    ],
    [
        {
            "text": "What's the latest price of Bitcoin?",
        }
    ],
    [
        {
            "text": "Search and give me list of spaces trending on HuggingFace.",
        }
    ],
    [
        {
            "text": "Create a Beautiful Picture of Effiel at Night.",
        }
    ],
    [
        {
            "text": "Create image of cute cat.",
        }
    ],
    [
        {
            "text": "What unusual happens in this video.",
            "files": [f"{examples_path}/example_video/accident.gif"],
        }
    ],
    [
        {
            "text": "What's name of superhero in this clip",
            "files": [f"{examples_path}/example_video/spiderman.gif"],
        }
    ],
    [
        {
            "text": "What's written on this paper",
            "files": [f"{examples_path}/example_images/paper_with_text.png"],
        }
    ],
    [
        {
            "text": "Who are they? Tell me about both of them",
            "files": [f"{examples_path}/example_images/elon_smoking.jpg",
                      f"{examples_path}/example_images/steve_jobs.jpg", ]
        }
    ]
]

# Set bot avatar image
BOT_AVATAR = "OpenAI_logo.png"

# Perform a Google search and return the results
@lru_cache(maxsize=128) 
def extract_text_from_webpage(html_content):
    """Extracts visible text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "form", "svg"]):
        tag.extract()
    visible_text = soup.get_text(strip=True)
    return visible_text

# Perform a Google search and return the results
def search(query):
    term = query
    start = 0
    all_results = []
    max_chars_per_page = 6000
    with requests.Session() as session:
        resp = session.get(
            url="https://www.google.com/search",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"},
            params={"q": term, "num": 4, "udm": 14},
            timeout=5,
            verify=None,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        result_block = soup.find_all("div", attrs={"class": "g"})
        for result in result_block:
            link = result.find("a", href=True)
            link = link["href"]
            try:
                webpage = session.get(link, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"}, timeout=5, verify=False)
                webpage.raise_for_status()
                visible_text = extract_text_from_webpage(webpage.text)
                if len(visible_text) > max_chars_per_page:
                    visible_text = visible_text[:max_chars_per_page]
                all_results.append({"link": link, "text": visible_text})
            except requests.exceptions.RequestException:
                all_results.append({"link": link, "text": None})
    return all_results


def image_gen(prompt):
    client = Client("KingNish/Image-Gen-Pro")
    return client.predict("Image Generation",None, prompt, api_name="/image_gen_pro")

def video_gen(prompt):
    client = Client("KingNish/Instant-Video")
    return client.predict(prompt, api_name="/instant_video")

def llava(user_prompt, chat_history):
    if user_prompt["files"]:
        image = user_prompt["files"][0]
    else:
        for hist in chat_history:
            if type(hist[0])==tuple:
                image = hist[0][0]
    
    txt = user_prompt["text"]
    img = user_prompt["files"]
    
    video_extensions = ("avi", "mp4", "mov", "mkv", "flv", "wmv", "mjpeg", "wav", "gif", "webm", "m4v", "3gp")
    image_extensions = Image.registered_extensions()
    image_extensions = tuple([ex for ex, f in image_extensions.items()])
        
    if image.endswith(video_extensions):
        image = sample_frames(image)
        gr.Info("Analyzing Video")
        image_tokens = "<image>" * int(len(image))
        prompt = f"<|im_start|>user {image_tokens}\n{user_prompt}<|im_end|><|im_start|>assistant"
          
    elif image.endswith(image_extensions):
        image = Image.open(image).convert("RGB")
        gr.Info("Analyzing image")
        prompt = f"<|im_start|>user <image>\n{user_prompt}<|im_end|><|im_start|>assistant"

    system_llava = "<|im_start|>system\nYou are OpenGPT 4o, an exceptionally capable and versatile AI assistant made by KingNish. Your task is to fulfill users query in best possible way. You are provided with image, videos and 3d structures as input with question your task is to give best possible detailed results to user according to their query. Reply the question asked by user properly and best possible way.<|im_end|>"
    
    final_prompt = f"{system_llava}\n{prompt}"
        
    inputs = processor(final_prompt, image, return_tensors="pt").to("cuda", torch.float16)

    return inputs

# Initialize inference clients for different models
client_mistral = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3")
client_mixtral = InferenceClient("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
client_llama = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
client_mistral_nemo = InferenceClient("mistralai/Mistral-Nemo-Instruct-2407")

@spaces.GPU(duration=60, queue=False)
def model_inference( user_prompt, chat_history):
    if user_prompt["files"]:
        inputs = llava(user_prompt, chat_history)
        streamer = TextIteratorStreamer(processor, skip_prompt=True, **{"skip_special_tokens": True})
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=2048)
    
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
    
        buffer = ""
        for new_text in streamer:
            buffer += new_text
            yield buffer

    else: 
        func_caller = []
        message = user_prompt

        functions_metadata = [
            {"type": "function", "function": {"name": "web_search", "description": "Search query on google and find latest information, info about any person, object, place thing, everything that available on google.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "web search query"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "general_query", "description": "Reply general query of USER, with LLM like you. But it does not answer tough questions and latest info's.", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A detailed prompt"}}, "required": ["prompt"]}}},
            {"type": "function", "function": {"name": "hard_query", "description": "Reply tough query of USER, using powerful LLM. But it does not answer latest info's.", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A detailed prompt"}}, "required": ["prompt"]}}},
            {"type": "function", "function": {"name": "image_generation", "description": "Generate image for user", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "image generation prompt"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "video_generation", "description": "Generate video for user", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "video generation prompt"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "image_qna", "description": "Answer question asked by user related to image", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Question by user"}}, "required": ["query"]}}},
        ]

        for msg in chat_history:
            func_caller.append({"role": "user", "content": f"{str(msg[0])}"})
            func_caller.append({"role": "assistant", "content": f"{str(msg[1])}"})

        message_text = message["text"]
        func_caller.append({"role": "user", "content": f'[SYSTEM]You are a helpful assistant. You have access to the following functions: \n {str(functions_metadata)}\n\nTo use these functions respond with:\n<functioncall> {{ "name": "function_name", "arguments": {{ "arg_1": "value_1", "arg_1": "value_1", ... }} }}  </functioncall>  [USER] {message_text}'})
    
        response = client_mistral.chat_completion(func_caller, max_tokens=200)
        response = str(response)
        try:
            response = response[response.find("{"):response.index("</")]
        except:
            response = response[response.find("{"):(response.rfind("}")+1)]
        response = response.replace("\\n", "")
        response = response.replace("\\'", "'")
        response = response.replace('\\"', '"')
        response = response.replace('\\', '')
        print(f"\n{response}")
    
        try:
            json_data = json.loads(str(response))
            if json_data["name"] == "web_search":
                query = json_data["arguments"]["query"]

                gr.Info("Searching Web")
                yield "Searching Web"
                web_results = search(query)
                
                gr.Info("Extracting relevant Info")
                yield "Extracting Relevant Info"
                web2 = ' '.join([f"Link: {res['link']}\nText: {res['text']}\n\n" for res in web_results])

                try:
                    message_groq = []
                    message_groq.append({"role":"system", "content": "You are OpenGPT 4o a helpful and powerful assistant made by KingNish. a helpful and very powerful chatbot web assistant made by KingNish. You are provided with WEB results from which you can find informations to answer users query in Structured, Deatailed and Better way, in Human Style. You are also Expert in every field and also learn and try to answer from contexts related to previous question. Try your best to give best response possible to user. You also try to show emotions using Emojis and reply in detail like human, use short forms, structured format, friendly tone and emotions."})
                    for msg in chat_history:
                        message_groq.append({"role": "user", "content": f"{str(msg[0])}"})
                        message_groq.append({"role": "assistant", "content": f"{str(msg[1])}"})
                    message_groq.append({"role": "user", "content": f"[USER] {str(message_text)} , [WEB RESULTS] {str(web2)}"})
                    # its meta-llama/Meta-Llama-3.1-8B-Instruct
                    stream = client_groq.chat.completions.create(model="llama-3.1-8b-instant",  messages=message_groq, max_tokens=4096, stream=True)
                    output = ""
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            output += chunk.choices[0].delta.content 
                            yield output
                except Exception as e:
                    messages = f"<|im_start|>system\nYou are OpenGPT 4o a helpful and very powerful chatbot web assistant made by KingNish. You are provided with WEB results from which you can find informations to answer users query in Structured, Better and in Human Way. You do not say Unnecesarry things. You are also Expert in every field and also learn and try to answer from contexts related to previous question. Try your best to give best response possible to user. You also try to show emotions using Emojis and reply in details like human, use short forms, friendly tone and emotions.<|im_end|>"
                    for msg in chat_history:
                        messages += f"\n<|im_start|>user\n{str(msg[0])}<|im_end|>"
                        messages += f"\n<|im_start|>assistant\n{str(msg[1])}<|im_end|>"
                    messages+=f"\n<|im_start|>user\n{message_text}<|im_end|>\n<|im_start|>web_result\n{web2}<|im_end|>\n<|im_start|>assistant\n"
                    
                    stream = client_mixtral.text_generation(messages, max_new_tokens=4000, do_sample=True, stream=True, details=True, return_full_text=False)
                    output = ""
                    for response in stream:
                        if not response.token.text == "<|im_end|>":
                            output += response.token.text
                            yield output
            
            elif json_data["name"] == "image_generation":
                query = json_data["arguments"]["query"]
                gr.Info("Generating Image, Please wait 10 sec...")
                yield "Generating Image, Please wait 10 sec..."
                image = image_gen(f"{str(query)}")
                yield gr.Image(image[1])

            elif json_data["name"] == "video_generation":
                query = json_data["arguments"]["query"]
                gr.Info("Generating Video, Please wait 15 sec...")
                yield "Generating Video, Please wait 15 sec..."
                video = video_gen(f"{str(query)}")
                yield gr.Video(video)
                
            elif json_data["name"] == "image_qna":
                inputs = llava(user_prompt, chat_history)
                streamer = TextIteratorStreamer(processor, skip_prompt=True, **{"skip_special_tokens": True})
                generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)

                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
    
                buffer = ""
                for new_text in streamer:
                    buffer += new_text
                    yield buffer

            elif json_data["name"] == "hard_query":
                try:
                    message_groq = []
                    message_groq.append({"role":"system", "content": "You are OpenGPT 4o a helpful and powerful assistant made by KingNish. You answers users query in detail and structured format and style like human. You are also Expert in every field and also learn and try to answer from contexts related to previous question. You also try to show emotions using Emojis and reply like human, use short forms, structured manner, detailed explaination, friendly tone and emotions."})
                    for msg in chat_history:
                        message_groq.append({"role": "user", "content": f"{str(msg[0])}"})
                        message_groq.append({"role": "assistant", "content": f"{str(msg[1])}"})
                    message_groq.append({"role": "user", "content": f"{str(message_text)}"})
                    # its meta-llama/Meta-Llama-3.1-70B-Instruct
                    stream = client_groq.chat.completions.create(model="llama-3.1-70b-versatile",  messages=message_groq, max_tokens=4096, stream=True)
                    output = ""
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            output += chunk.choices[0].delta.content 
                            yield output
                except Exception as e:
                    print(e)
                    try:
                        message_groq = []
                        message_groq.append({"role":"system", "content": "You are OpenGPT 4o a helpful and powerful assistant made by KingNish. You answers users query in detail and structured format and style like human. You are also Expert in every field and also learn and try to answer from contexts related to previous question. You also try to show emotions using Emojis and reply like human, use short forms, structured manner, detailed explaination, friendly tone and emotions."})
                        for msg in chat_history:
                            message_groq.append({"role": "user", "content": f"{str(msg[0])}"})
                            message_groq.append({"role": "assistant", "content": f"{str(msg[1])}"})
                        message_groq.append({"role": "user", "content": f"{str(message_text)}"})
                        # its meta-llama/Meta-Llama-3-70B-Instruct
                        stream = client_groq.chat.completions.create(model="llama3-70b-8192",  messages=message_groq, max_tokens=4096, stream=True)
                        output = ""
                        for chunk in stream:
                            content = chunk.choices[0].delta.content
                            if content:
                                output += chunk.choices[0].delta.content 
                                yield output
                    except Exception as e:
                        print(e)
                        message_groq = []
                        message_groq.append({"role":"system", "content": "You are OpenGPT 4o a helpful and powerful assistant made by KingNish. You answers users query in detail and structured format and style like human. You are also Expert in every field and also learn and try to answer from contexts related to previous question. You also try to show emotions using Emojis and reply like human, use short forms, structured manner, detailed explaination, friendly tone and emotions."})
                        for msg in chat_history:
                            message_groq.append({"role": "user", "content": f"{str(msg[0])}"})
                            message_groq.append({"role": "assistant", "content": f"{str(msg[1])}"})
                        message_groq.append({"role": "user", "content": f"{str(message_text)}"})
                        stream = client_groq.chat.completions.create(model="llama3-groq-70b-8192-tool-use-preview",  messages=message_groq, max_tokens=4096, stream=True)
                        output = ""
                        for chunk in stream:
                            content = chunk.choices[0].delta.content
                            if content:
                                output += chunk.choices[0].delta.content 
                                yield output
            else:
                try:
                    message_groq = []
                    message_groq.append({"role":"system", "content": "You are OpenGPT 4o a helpful and powerful assistant made by KingNish. You answers users query in detail and structured format and style like human. You are also Expert in every field and also learn and try to answer from contexts related to previous question. You also try to show emotions using Emojis and reply like human, use short forms, structured manner, detailed explaination, friendly tone and emotions."})
                    for msg in chat_history:
                        message_groq.append({"role": "user", "content": f"{str(msg[0])}"})
                        message_groq.append({"role": "assistant", "content": f"{str(msg[1])}"})
                    message_groq.append({"role": "user", "content": f"{str(message_text)}"})
                    # its meta-llama/Meta-Llama-3-70B-Instruct
                    stream = client_groq.chat.completions.create(model="llama3-70b-8192",  messages=message_groq, max_tokens=4096, stream=True)
                    output = ""
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            output += chunk.choices[0].delta.content 
                            yield output
                except Exception as e:
                    print(e)
                    try:
                        message_groq = []
                        message_groq.append({"role":"system", "content": "You are OpenGPT 4o a helpful and powerful assistant made by KingNish. You answers users query in detail and structured format and style like human. You are also Expert in every field and also learn and try to answer from contexts related to previous question. You also try to show emotions using Emojis and reply like human, use short forms, structured manner, detailed explaination, friendly tone and emotions."})
                        for msg in chat_history:
                            message_groq.append({"role": "user", "content": f"{str(msg[0])}"})
                            message_groq.append({"role": "assistant", "content": f"{str(msg[1])}"})
                        message_groq.append({"role": "user", "content": f"{str(message_text)}"})
                        # its meta-llama/Meta-Llama-3-8B-Instruct
                        stream = client_groq.chat.completions.create(model="llama3-8b-8192",  messages=message_groq, max_tokens=4096, stream=True)
                        output = ""
                        for chunk in stream:
                            content = chunk.choices[0].delta.content
                            if content:
                                output += chunk.choices[0].delta.content 
                                yield output
                    except Exception as e:
                        print(e)
                        messages = f"<|start_header_id|>system\nYou are OpenGPT 4o a helpful and powerful assistant made by KingNish. You answers users query in detail and structured format and style like human. You are also Expert in every field and also learn and try to answer from contexts related to previous question. You also try to show emotions using Emojis and reply like human, use short forms, structured manner, detailed explaination, friendly tone and emotions.<|end_header_id|>"
                        for msg in chat_history:
                            messages += f"\n<|start_header_id|>user\n{str(msg[0])}<|end_header_id|>"
                            messages += f"\n<|start_header_id|>assistant\n{str(msg[1])}<|end_header_id|>"
                        messages+=f"\n<|start_header_id|>user\n{message_text}<|end_header_id|>\n<|start_header_id|>assistant\n"
                        stream = client_llama.text_generation(messages, max_new_tokens=2000, do_sample=True, stream=True, details=True, return_full_text=False)
                        output = ""
                        for response in stream:
                            if not response.token.text == "<|eot_id|>":
                                output += response.token.text
                                yield output
        except Exception as e:
            print(e)
            try:
                message_groq = []
                message_groq.append({"role":"system", "content": "You are OpenGPT 4o a helpful and powerful assistant made by KingNish. You answers users query in detail and structured format and style like human. You are also Expert in every field and also learn and try to answer from contexts related to previous question. You also try to show emotions using Emojis and reply like human, use short forms, structured manner, detailed explaination, friendly tone and emotions."})
                for msg in chat_history:
                    message_groq.append({"role": "user", "content": f"{str(msg[0])}"})
                    message_groq.append({"role": "assistant", "content": f"{str(msg[1])}"})
                message_groq.append({"role": "user", "content": f"{str(message_text)}"})
                # its meta-llama/Meta-Llama-3-70B-Instruct
                stream = client_groq.chat.completions.create(model="llama3-70b-8192",  messages=message_groq, max_tokens=4096, stream=True)
                output = ""
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        output += chunk.choices[0].delta.content 
                        yield output
            except Exception as e:
                print(e)
                try:
                    message_groq = []
                    message_groq.append({"role":"system", "content": "You are OpenGPT 4o a helpful and powerful assistant made by KingNish. You answers users query in detail and structured format and style like human. You are also Expert in every field and also learn and try to answer from contexts related to previous question. You also try to show emotions using Emojis and reply like human, use short forms, structured manner, detailed explaination, friendly tone and emotions."})
                    for msg in chat_history:
                        message_groq.append({"role": "user", "content": f"{str(msg[0])}"})
                        message_groq.append({"role": "assistant", "content": f"{str(msg[1])}"})
                    message_groq.append({"role": "user", "content": f"{str(message_text)}"})
                    # its meta-llama/Meta-Llama-3-8B-Instruct
                    stream = client_groq.chat.completions.create(model="llama3-8b-8192",  messages=message_groq, max_tokens=4096, stream=True)
                    output = ""
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            output += chunk.choices[0].delta.content 
                            yield output
                except Exception as e:
                    print(e)
                    messages = f"<|im_start|>system\nYou are OpenGPT 4o a helpful and powerful assistant made by KingNish. You answers users query in detail and structured format and style like human. You are also Expert in every field and also learn and try to answer from contexts related to previous question. You also try to show emotions using Emojis and reply like human, use short forms, structured manner, detailed explaination, friendly tone and emotions.<|im_end|>"
                    for msg in chat_history:
                        messages += f"\n<|im_start|>user\n{str(msg[0])}<|im_end|>"
                        messages += f"\n<|im_start|>assistant\n{str(msg[1])}<|im_end|>"
                    messages+=f"\n<|im_start|>user\n{message_text}<|im_end|>\n<|im_start|>assistant\n"
                    stream = client_mixtral.text_generation(messages, max_new_tokens=4000, do_sample=True, stream=True, details=True, return_full_text=False)
                    output = ""
                    for response in stream:
                        if not response.token.text == "<|im_end|>":
                            output += response.token.text
                            yield output
                    
# Create a chatbot interface
chatbot = gr.Chatbot(
    label="OpenGPT-4o",
    avatar_images=[None, BOT_AVATAR],
    show_copy_button=True,
    likeable=True,
    layout="panel",
    height=400,
)
output = gr.Textbox(label="Prompt")
