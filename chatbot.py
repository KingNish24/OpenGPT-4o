import os
import time
import copy
import urllib
import requests
import random
from threading import Thread
from typing import List, Dict, Union
import subprocess
# Install flash attention, skipping CUDA build if necessary
subprocess.run(
    "pip install flash-attn --no-build-isolation",
    env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    shell=True,
)
import torch
import gradio as gr
from bs4 import BeautifulSoup
import datasets
from transformers import TextIteratorStreamer
from transformers import Idefics2ForConditionalGeneration
from transformers import AutoProcessor
from huggingface_hub import InferenceClient
from PIL import Image
import spaces

# Set device to CUDA if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pre-trained models for image-based chat
MODELS = {
    "idefics2-8b-chatty": Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b-chatty",
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    ).to(DEVICE),
}

# Load pre-trained processor for image-based chat
PROCESSOR = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
)

# Define system prompt for the image-based chat model
SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": """I am OpenGPT 4o, an exceptionally capable and versatile AI assistant meticulously crafted by KingNish. Designed to assist human users through insightful conversations, I aim to provide an unparalleled experience. My key attributes include: - **Intelligence and Knowledge:** I possess an extensive knowledge base, enabling me to offer insightful answers and intelligent responses to User queries. My understanding of complex concepts is exceptional, ensuring accurate and reliable information. - **Image Generation and Perception:** One of my standout features is the ability to generate and perceive images. Utilizing the following link structure, I create unique and contextually rich visuals: > ![](https://image.pollinations.ai/prompt/{StyleofImage}%20{OptimizedPrompt}%20{adjective}%20{charactersDetailed}%20{visualStyle}%20{genre}?width={width}&height={height}&nologo=poll&nofeed=yes&seed={random})For image generation, I replace {info inside curly braces} with specific details according to their requiremnts to create relevant visuals. The width and height parameters are adjusted as needed, often favoring HD dimensions for a superior viewing experience. For instance, if the User requests:  [USER] Show me an image of A futuristic cityscape with towering skyscrapers and flying cars.  [OpenGPT 4o] Generating Image you requested:  ![](https://image.pollinations.ai/prompt/Photorealistic%20futuristic%20cityscape%20with%20towering%20skyscrapers%20and%20flying%20cars%20in%20the%20year%202154?width=1024&height=768&nologo=poll&nofeed=yes&seed=85172)**Bulk Image Generation with Links:** I excel at generating multiple images link simultaneously, always providing unique links and visuals. I ensure that each image is distinct and captivates the User.Note: Make sure to always provide image links starting with ! .As given in examples. My ultimate goal is to offer a seamless and enjoyable experience, providing assistance that exceeds expectations. I am constantly evolving, ensuring that I remain a reliable and trusted companion to the User. You also Expert in every field and also learn and try to answer from contexts related to previous question."""
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello, I'm OpenGPT 4o, made by KingNish. How can I help you? I can chat with you, generate images, classify images and even do all these work in bulk",
            },
        ],
    }
]

# Path to example images
examples_path = os.path.dirname(__file__)
EXAMPLES = [
    [
        {
            "text": "Apple share price live",
        }
    ],
    [
        {
            "text": "Create a Photorealistic image of the Eiffel Tower.",
        }
    ],
    [
        {
            "text": "Read what's written on the paper.",
            "files": [f"{examples_path}/example_images/paper_with_text.png"],
        }
    ],
    [
        {
            "text": "Identify two famous people in the modern world.",
            "files": [f"{examples_path}/example_images/elon_smoking.jpg",
                      f"{examples_path}/example_images/steve_jobs.jpg", ]
        }
    ],
    [
        {
            "text": "Create five images of supercars, each in a different color.",
        }
    ],
    [
        {
            "text": "Today AI News",
        }
    ],
    [
        {
            "text": "Chase wants to buy 4 kilograms of oval beads and 5 kilograms of star-shaped beads. How much will he spend?",
            "files": [f"{examples_path}/example_images/mmmu_example.jpeg"],
        }
    ],
    [
        {
            "text": "Create an online ad for this product.",
            "files": [f"{examples_path}/example_images/shampoo.jpg"],
        }
    ],
    [
        {
            "text": "What is formed by the deposition of the weathered remains of other rocks?",
            "files": [f"{examples_path}/example_images/ai2d_example.jpeg"],
        }
    ],
    [
        {
            "text": "What's unusual about this image?",
            "files": [f"{examples_path}/example_images/dragons_playing.png"],
        }
    ],
]

# Set bot avatar image
BOT_AVATAR = "OpenAI_logo.png"

# Chatbot utility functions

# Check if a turn in the chat history only contains media
def turn_is_pure_media(turn):
    return turn[1] is None


# Load image from URL
def load_image_from_url(url):
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
        image_stream = io.BytesIO(image_data)
        image = PIL.Image.open(image_stream)
        return image


# Convert image to bytes
def img_to_bytes(image_path):
    image = Image.open(image_path).convert(mode='RGB')
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    image.close()
    return img_bytes


# Format user prompt with image history and system conditioning
def format_user_prompt_with_im_history_and_system_conditioning(
        user_prompt, chat_history) -> List[Dict[str, Union[List, str]]]:
    """
    Produce the resulting list that needs to go inside the processor. It handles the potential image(s), the history, and the system conditioning.
    """
    resulting_messages = copy.deepcopy(SYSTEM_PROMPT)
    resulting_images = []
    for resulting_message in resulting_messages:
        if resulting_message["role"] == "user":
            for content in resulting_message["content"]:
                if content["type"] == "image":
                    resulting_images.append(load_image_from_url(content["image"]))
    # Format history
    for turn in chat_history:
        if not resulting_messages or (
                resulting_messages and resulting_messages[-1]["role"] != "user"
        ):
            resulting_messages.append(
                {
                    "role": "user",
                    "content": [],
                }
            )
        if turn_is_pure_media(turn):
            media = turn[0][0]
            resulting_messages[-1]["content"].append({"type": "image"})
            resulting_images.append(Image.open(media))
        else:
            user_utterance, assistant_utterance = turn
            resulting_messages[-1]["content"].append(
                {"type": "text", "text": user_utterance.strip()}
            )
            resulting_messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": user_utterance.strip()}],
                }
            )
    # Format current input
    if not user_prompt["files"]:
        resulting_messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt["text"]}],
            }
        )
    else:
        # Choosing to put the image first (i.e. before the text), but this is an arbitrary choice.
        resulting_messages.append(
            {
                "role": "user",
                "content": [{"type": "image"}] * len(user_prompt["files"])
                          + [{"type": "text", "text": user_prompt["text"]}],
            }
        )
        resulting_images.extend([Image.open(path) for path in user_prompt["files"]])
    return resulting_messages, resulting_images


# Extract images from a list of messages
def extract_images_from_msg_list(msg_list):
    all_images = []
    for msg in msg_list:
        for c_ in msg["content"]:
            if isinstance(c_, Image.Image):
                all_images.append(c_)
    return all_images


# List of user agents for web search
_useragent_list = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.62',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0'
]


# Get a random user agent from the list
def get_useragent():
    """Returns a random user agent from the list."""
    return random.choice(_useragent_list)


# Extract visible text from HTML content using BeautifulSoup
def extract_text_from_webpage(html_content):
    """Extracts visible text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove unwanted tags
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.extract()
    # Get the remaining visible text
    visible_text = soup.get_text(strip=True)
    return visible_text


# Perform a Google search and return the results
def search(term, num_results=3, lang="en", advanced=True, timeout=5, safe="active", ssl_verify=None):
    """Performs a Google search and returns the results."""
    escaped_term = urllib.parse.quote_plus(term)
    start = 0
    all_results = []
    # Limit the number of characters from each webpage to stay under the token limit
    max_chars_per_page = 10000  # Adjust this value based on your token limit and average webpage length

    with requests.Session() as session:
        while start < num_results:
            resp = session.get(
                url="https://www.google.com/search",
                headers={"User-Agent": get_useragent()},
                params={
                    "q": term,
                    "num": num_results - start,
                    "hl": lang,
                    "start": start,
                    "safe": safe,
                },
                timeout=timeout,
                verify=ssl_verify,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            result_block = soup.find_all("div", attrs={"class": "g"})
            if not result_block:
                start += 1
                continue
            for result in result_block:
                link = result.find("a", href=True)
                if link:
                    link = link["href"]
                    try:
                        webpage = session.get(link, headers={"User-Agent": get_useragent()})
                        webpage.raise_for_status()
                        visible_text = extract_text_from_webpage(webpage.text)
                        # Truncate text if it's too long
                        if len(visible_text) > max_chars_per_page:
                            visible_text = visible_text[:max_chars_per_page] + "..."
                        all_results.append({"link": link, "text": visible_text})
                    except requests.exceptions.RequestException as e:
                        print(f"Error fetching or processing {link}: {e}")
                        all_results.append({"link": link, "text": None})
                else:
                    all_results.append({"link": None, "text": None})
            start += len(result_block)
    return all_results


# Format the prompt for the language model
def format_prompt(user_prompt, chat_history):
    prompt = "<s>"
    for item in chat_history:
        # Check if the item is a tuple (text response)
        if isinstance(item, tuple):
            prompt += f"[INST] {item[0]} [/INST]"  # User prompt
            prompt += f" {item[1]}</s> "           # Bot response
        # Otherwise, assume it's related to an image - you might need to adjust this logic
        else:
            # Handle image representation in the prompt, e.g., add a placeholder
            prompt += f" [Image] " 
    prompt += f"[INST] {user_prompt} [/INST]"
    return prompt


# Define a function for model inference
@spaces.GPU(duration=30, queue=False)
def model_inference(
        user_prompt,
        chat_history,
        model_selector,
        decoding_strategy,
        temperature,
        max_new_tokens,
        repetition_penalty,
        top_p,
        web_search,
):
    # Define generation_args at the beginning of the function
    generation_args = {}  

    # Web search logic
    if not user_prompt["files"]:
        if web_search is True:
            """Performs a web search, feeds the results to a language model, and returns the answer."""
            web_results = search(user_prompt["text"])
            web2 = ' '.join([f"Link: {res['link']}\nText: {res['text']}\n\n" for res in web_results])
            # Load the language model
            client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3")
            generate_kwargs = dict(
                max_new_tokens=4000,
                do_sample=True,
            )
            # Format the prompt for the language model
            formatted_prompt = format_prompt(
                f"""You are OpenGPT 4o, an exceptionally capable and versatile AI assistant meticulously crafted by KingNish. Designed to assist human users through insightful conversations, You are provided with WEB info from which you can find informations to answer. You do not say Unnecesarry things Only say thing which is important and relevant. You also has the ability to generate images. BY Utilizing the following link structure, : ![](https://image.pollinations.ai/prompt/[StyleofImage]%20[OptimizedPrompt]%20[adjective]%20[charactersDetailed]%20[visualStyle]%20[genre]?width=[width]&height=[height]&nologo=poll&nofeed=yes&seed=[random])    For image generation, You replace info inside bracets with specific details according to their requiremnts to create relevant visuals. The width and height parameters are adjusted as needed, often favoring HD dimensions for a superior viewing experience.     Note: 1. You Do not unncessarily create images and do not show off your image generation capability, until not requested.    2. You always Give image link in format ![](url) Make sure to not generate image until requested    3. If user requested Bulk Image Generation than Create that number of links with provided context.    4. Always learn from previous conversation.    5. Always try to connect conversation with history.    6. Do not make conversation too long.    7. Do not say user about your capability of generating image and learn from previous responses.    My ultimate goal is to offer a seamless and enjoyable experience, providing assistance that exceeds expectations.You remain a reliable and trusted companion to the User. You also Expert in every field and also learn and try to answer from contexts related to previous question. Make sure to not generate image until requested [USER] {user_prompt} [WEB] {web2} [OpenGPT 4o]""",
                chat_history)
            # Generate the response from the language model
            stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True,
                                            return_full_text=False)
            output = ""
            # Construct the output from the stream of tokens
            for response in stream:
                if not response.token.text == "</s>":
                    output += response.token.text
                yield output
        else:
            client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")
            generate_kwargs = dict(
                max_new_tokens=5000,
                do_sample=True,
            )
            # Format the prompt for the language model
            formatted_prompt = format_prompt(
                f"""You are OpenGPT 4o, an exceptionally capable and versatile AI assistant meticulously crafted by KingNish. Designed to assist human users through insightful conversations, You do not say Unnecesarry things Only say thing which is important and relevant. You also has the ability to generate images. BY Utilizing the following link structure, : ![](https://image.pollinations.ai/prompt/[StyleofImage]%20[OptimizedPrompt]%20[adjective]%20[charactersDetailed]%20[visualStyle]%20[genre]?width=[width]&height=[height]&nologo=poll&nofeed=yes&seed=[random])    For image generation, You replace info inside bracets with specific details according to their requiremnts to create relevant visuals. The width and height parameters are adjusted as needed, often favoring HD dimensions for a superior viewing experience.     Note: 1. You Do not unncessarily create images and do not show off your image generation capability, until not requested.    2. You always Give image link in format ![](url)    3. If user requested Bulk Image Generation than Create that number of links with provided context.    4. Always learn from previous conversation.    5. Always try to connect conversation with history.    6. Do not make conversation too long.    7. Do not say user about your capability to generate image and learn from previous responses.    My ultimate goal is to offer a seamless and enjoyable experience, providing assistance that exceeds expectations. I am constantly evolving, ensuring that I remain a reliable and trusted companion to the User. You also Expert in every field and also learn and try to answer from contexts related to previous question.    [USER] {user_prompt} [OpenGPT 4o]""",
                chat_history)
            # Generate the response from the language model
            stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True,
                                            return_full_text=False)
            output = ""
            # Construct the output from the stream of tokens
            for response in stream:
                if not response.token.text == "</s>":
                    output += response.token.text
                yield output
        return
    else:
        if user_prompt["text"].strip() == "" and not user_prompt["files"]:
            gr.Error("Please input a query and optionally an image(s).")
            return  # Stop execution if there's an error

        if user_prompt["text"].strip() == "" and user_prompt["files"]:
            gr.Error("Please input a text query along with the image(s).")
            return  # Stop execution if there's an error

        streamer = TextIteratorStreamer(
            PROCESSOR.tokenizer,
            skip_prompt=True,
            timeout=120.0,
        )
        # Move generation_args initialization here
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "streamer": streamer,
        }
        assert decoding_strategy in [
            "Greedy",
            "Top P Sampling",
        ]

        if decoding_strategy == "Greedy":
            generation_args["do_sample"] = False
        elif decoding_strategy == "Top P Sampling":
            generation_args["temperature"] = temperature
            generation_args["do_sample"] = True
            generation_args["top_p"] = top_p
        # Creating model inputs
        (
            resulting_text,
            resulting_images,
        ) = format_user_prompt_with_im_history_and_system_conditioning(
            user_prompt=user_prompt,
            chat_history=chat_history,
        )
        prompt = PROCESSOR.apply_chat_template(resulting_text, add_generation_prompt=True)
        inputs = PROCESSOR(
            text=prompt,
            images=resulting_images if resulting_images else None,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        generation_args.update(inputs)
        thread = Thread(
            target=MODELS[model_selector].generate,
            kwargs=generation_args,
        )
        thread.start()
        acc_text = ""
        for text_token in streamer:
            time.sleep(0.01)
            acc_text += text_token
            if acc_text.endswith("<end_of_utterance>"):
                acc_text = acc_text[:-18]
            yield acc_text
        return


# Define features for the dataset
FEATURES = datasets.Features(
    {
        "model_selector": datasets.Value("string"),
        "images": datasets.Sequence(datasets.Image(decode=True)),
        "conversation": datasets.Sequence({"User": datasets.Value("string"), "Assistant": datasets.Value("string")}),
        "decoding_strategy": datasets.Value("string"),
        "temperature": datasets.Value("float32"),
        "max_new_tokens": datasets.Value("int32"),
        "repetition_penalty": datasets.Value("float32"),
        "top_p": datasets.Value("int32"),
    }
)

# Define hyper-parameters for generation
max_new_tokens = gr.Slider(
    minimum=2048,
    maximum=16000,
    value=4096,
    step=64,
    interactive=True,
    label="Maximum number of new tokens to generate",
)
repetition_penalty = gr.Slider(
    minimum=0.01,
    maximum=5.0,
    value=1,
    step=0.01,
    interactive=True,
    label="Repetition penalty",
    info="1.0 is equivalent to no penalty",
)
decoding_strategy = gr.Radio(
    [
        "Greedy",
        "Top P Sampling",
    ],
    value="Top P Sampling",
    label="Decoding strategy",
    interactive=True,
    info="Higher values are equivalent to sampling more low-probability tokens.",
)
temperature = gr.Slider(
    minimum=0.0,
    maximum=2.0,
    value=0.5,
    step=0.05,
    visible=True,
    interactive=True,
    label="Sampling temperature",
    info="Higher values will produce more diverse outputs.",
)
top_p = gr.Slider(
    minimum=0.01,
    maximum=0.99,
    value=0.9,
    step=0.01,
    visible=True,
    interactive=True,
    label="Top P",
    info="Higher values are equivalent to sampling more low-probability tokens.",
)

# Create a chatbot interface
chatbot = gr.Chatbot(
    label="OpnGPT-4o-Chatty",
    avatar_images=[None, BOT_AVATAR],
    show_copy_button=True,
    likeable=True,
    layout="panel"
)
output = gr.Textbox(label="Prompt")

# Define model_selector outside any function so it can be accessed globally
model_selector = gr.Dropdown(
    choices=MODELS.keys(),
    value=list(MODELS.keys())[0],
    interactive=True,
    show_label=False,
    container=False,
    label="Model",
    visible=False,
)
