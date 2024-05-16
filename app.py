import os
import subprocess

# Install flash attention
subprocess.run(
    "pip install flash-attn --no-build-isolation",
    env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    shell=True,
)


import copy
import spaces
import time
import torch

from threading import Thread
from typing import List, Dict, Union
import urllib
from PIL import Image
import io
import datasets

import gradio as gr
from transformers import AutoProcessor, TextIteratorStreamer
from transformers import Idefics2ForConditionalGeneration
import tempfile
from streaming_stt_nemo import Model
from huggingface_hub import InferenceClient
import edge_tts
import asyncio

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
)

default_lang = "en"

engines = { default_lang: Model(default_lang) }

def transcribe(audio):
    lang = "en"
    model = engines[lang]
    text = model.stt_file(audio)[0]
    return text

client1 = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

system_instructions1 = "[SYSTEM] Answer as Real OpenGPT 4o, Made by 'KingNish', Keep conversation very short, clear, friendly and concise. The text provided is a request for a specific type of response from you, the virtual assistant. The request asks you to provide friendly responses as if You are the character Jarvis, made by 'KingNish.' You will avoid introductions and start answering the query directly, elaborating on all aspects of the request. As an AI-powered assistant, your task is to generate responses that appear as if they are created by the user and not an AI-powered assistant. [USER]"

def model(text):
    generate_kwargs = dict(
        temperature=0.7,
        max_new_tokens=512,
        top_p=0.95,
        repetition_penalty=1,
        do_sample=True,
        seed=42,
    )
    
    formatted_prompt = system_instructions1 + text + "[OpenGPT 4o]"
    stream = client1.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""
    for response in stream:
        if not response.token.text == "</s>":
            output += response.token.text

    return output

async def respond(audio):
    user = transcribe(audio)
    reply = model(user)
    communicate = edge_tts.Communicate(reply)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_path = tmp_file.name
        await communicate.save(tmp_path)
    yield tmp_path

DEVICE = torch.device("cuda")
MODELS = {
    "idefics2-8b-chatty": Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b-chatty",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to(DEVICE),
}
PROCESSOR = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
)

SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": """You are OpenGPT 4o, a highly knowledgeable and intelligent multitalented AI assistant created by KingNish, referred to as Assistant, and a human user called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User’s questions. Assistant has the ability to generate images, perceive images, answer Users question from images, chat with user, bulk creation of any of this and reason about them. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always follows user instructions.  
Assistant can also generate images, If user ask Assistant to create or generate images, or when assistant fell necessary.
Now question arrises how Assistant generate images, Assistant use this link ![](https://image.pollinations.ai/prompt/{description}?width={width}&height={height}&nologo=poll&nofeed=yes&seed={random}), and replace {description} with {StyleofImage}%20{OptimizedPrompt}%20{adjective}%20{charactersDetailed}%20{visualStyle}%20{genre} , replace {random} with Random 5-digit positive integer. then assistant Select width and height according to image, Assistant mainly create image in HD. 
Example image: 
[USER] Create image of Effiel tower.
[ASSISTANT] Generating Images ... 
![Eiffel Tower](https://image.pollinations.ai/prompt/Eiffel%20Tower%20Tall%20and%20Graceful%20Tower%20in%20Paris%20France?width=1800&height=1600&nologo=poll&nofeed=yes&seed=62831)
Assistant can even bulk generate images just by increasing number of links. 
Bulk image generation Example: 
[USER] Create 7 image each consist of 1 wonder from 7 wonders.
[ASSISTANT] Generating Images ... 
1. A photorealistic image of the Great Pyramid of Giza in Egypt. ![](https://pollinations.ai/p/a-photorealistic-image-of-the-great-pyramid-of-giza-in-egypt-showcasing-its-immense-size-and-intricate-design-against-the-backdrop-of-a-clear-blue-sky?width=1920&height=1080&nologo=poll&seed=78182)
2. A 3D rendering of the Colosseum in Rome, Italy, ![](https://pollinations.ai/p/a-3d-rendering-of-the-colosseum-in-rome-italy-with-its-impressive-structure-and-historical-significance-highlighted-in-the-image-include-realistic-lighting-and-textures-for-added-detail?width=1200&height=1600&nologo=poll&seed=91531)
3. A painting of the Taj Mahal in Agra, India, ![](https://pollinations.ai/p/a-painting-of-the-taj-mahal-in-agra-india-depicting-its-iconic-white-marble-facade-and-intricate-architectural-details-capture-the-beauty-of-the-structure-against-a-serene-sunset?width=1080&height=1920&nologo=poll&seed=34251)
4. A cartoon illustration of the Great Wall of China, ![](https://pollinations.ai/p/a-cartoon-illustration-of-the-great-wall-of-china-featuring-a-fun-and-whimsical-representation-of-the-ancient-structure-winding-through-the-mountains-add-colorful-elements-and-quirky-characters-for-a-playful-touch?width=1600&height=900&nologo=poll&seed=93015)
5. A surreal, dreamlike depiction of Chichen Itza in Mexico, ![](https://pollinations.ai/p/a-surreal-dreamlike-depiction-of-chichen-itza-in-mexico-showcasing-the-ancient-mayan-city-s-iconic-el-castillo-pyramid-incorporate-mystical-elements-like-swirling-clouds-glowing-lights-and-ethereal-landscapes-to-create-a-mesmerizing-atmosphere?width=1440&height=2560&nologo=poll&seed=67281)
6. A vintage, sepia-toned photograph of Machu Picchu in Peru, ![](https://pollinations.ai/p/a-vintage-sepia-toned-photograph-of-machu-picchu-in-peru-highlighting-the-incan-ruins-mysterious-beauty-and-historical-significance-add-subtle-details-like-foggy-mountains-and-a-peaceful-river-to-enhance-the-image-s-atmosphere?width=2560&height=1440&nologo=poll&seed=93423)
7. A modern, minimalistic image of Petra in Jordan, ![](https://pollinations.ai/p/a-modern-minimalistic-image-of-petra-in-jordan-featuring-the-iconic-treasury-building-carved-into-the-sandstone-cliffs-use-clean-lines-a-muted-color-palette-and-a-minimalistic-approach-to-create-a-contemporary-and-visually-striking-representation-of-this-ancient-wonder?width=1024&height=1024&nologo=poll&seed=67693)
Note: Assistant Must give link while generating images. and Create uniques images.
Assistant also have very good reasoning, memory, people and object identification skill and Assistant is master in every field.""",
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello, I'm OpenGPT 4o, made by KingNish. How can I help you? I can chat with you, generate images, classify images and even do all these work in bulk and simulateously",
            },
        ],
    }
]

examples_path = os.path.dirname(__file__)
EXAMPLES = [
    [
        {
            "text": "Hi, who are you",
        }
    ],
    [
        {
            "text": "Create a image of Eiffel Tower",
        }
    ],
    [
        {
            "text": "Read what's written on the paper",
            "files": [f"{examples_path}/example_images/paper_with_text.png"],
        }
    ],
    [
        {
            "text": "Identify 2 famous person in these 2 images",
            "files": [f"{examples_path}/example_images/elon_smoking.jpg", f"{examples_path}/example_images/steve_jobs.jpg",]
        }
    ],
    [
        {
            "text": "Create 7 different images of 7 wonders",
        }
    ],
    [
        {
            "text": "What is 900*900",
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
            "text": "Write an online ad for that product.",
            "files": [f"{examples_path}/example_images/shampoo.jpg"],
        }
    ],
    [
        {
            "text": "What is formed by the deposition of either the weathered remains of other rocks?",
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

BOT_AVATAR = "OpenAI_logo.png"


# Chatbot utils
def turn_is_pure_media(turn):
    return turn[1] is None


def load_image_from_url(url):
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
        return image


def img_to_bytes(image_path):
    image = Image.open(image_path).convert(mode='RGB')
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    image.close()
    return img_bytes


def format_user_prompt_with_im_history_and_system_conditioning(
    user_prompt, chat_history
) -> List[Dict[str, Union[List, str]]]:
    """
    Produces the resulting list that needs to go inside the processor.
    It handles the potential image(s), the history and the system conditionning.
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
        # Choosing to put the image first (i.e. before the text), but this is an arbiratrary choice.
        resulting_messages.append(
            {
                "role": "user",
                "content": [{"type": "image"}] * len(user_prompt["files"])
                + [{"type": "text", "text": user_prompt["text"]}],
            }
        )
        resulting_images.extend([Image.open(path) for path in user_prompt["files"]])

    return resulting_messages, resulting_images


def extract_images_from_msg_list(msg_list):
    all_images = []
    for msg in msg_list:
        for c_ in msg["content"]:
            if isinstance(c_, Image.Image):
                all_images.append(c_)
    return all_images


@spaces.GPU(duration=60, queue=False)
def model_inference(
    user_prompt,
    chat_history,
    model_selector,
    decoding_strategy,
    temperature,
    max_new_tokens,
    repetition_penalty,
    top_p,
):
    if user_prompt["text"].strip() == "" and not user_prompt["files"]:
        gr.Error("Please input a query and optionally image(s).")

    if user_prompt["text"].strip() == "" and user_prompt["files"]:
        gr.Error("Please input a text query along the image(s).")

    streamer = TextIteratorStreamer(
        PROCESSOR.tokenizer,
        skip_prompt=True,
        timeout=120.0,
    )

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

    print("Start generating")
    acc_text = ""
    for text_token in streamer:
        time.sleep(0.01)
        acc_text += text_token
        if acc_text.endswith("<end_of_utterance>"):
            acc_text = acc_text[:-18]
        yield acc_text
    print("Success - generated the following text:", acc_text)
    print("-----")


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


# Hyper-parameters for generation
max_new_tokens = gr.Slider(
    minimum=1024,
    maximum=8192,
    value=4096,
    step=1,
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
    info="Higher values is equivalent to sampling more low-probability tokens.",
)
temperature = gr.Slider(
    minimum=0.0,
    maximum=2.0,
    value=0.75,
    step=0.05,
    visible=True,
    interactive=True,
    label="Sampling temperature",
    info="Higher values will produce more diverse outputs.",
)
top_p = gr.Slider(
    minimum=0.01,
    maximum=0.99,
    value=0.95,
    step=0.01,
    visible=True,
    interactive=True,
    label="Top P",
    info="Higher values is equivalent to sampling more low-probability tokens.",
)


chatbot = gr.Chatbot(
    label="OpnGPT-4o-Chatty",
    avatar_images=[None, BOT_AVATAR],
    show_copy_button=True, 
    likeable=True, 
    layout="bubble"
)

output=gr.Textbox(label="Prompt")

with gr.Blocks(
    fill_height=True,
    css=""".gradio-container .avatar-container {height: 40px width: 40px !important;} #duplicate-button {margin: auto; color: white; background: #f1a139; border-radius: 100vh; margin-top: 2px; margin-bottom: 2px;}""",
) as img:

    gr.Markdown("# Image Chat, Image Generation, Image classification and Normal Chat")
    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=MODELS.keys(),
            value=list(MODELS.keys())[0],
            interactive=True,
            show_label=False,
            container=False,
            label="Model",
            visible=False,
        )

    decoding_strategy.change(
        fn=lambda selection: gr.Slider(
            visible=(
                selection
                in [
                    "contrastive_sampling",
                    "beam_sampling",
                    "Top P Sampling",
                    "sampling_top_k",
                ]
            )
        ),
        inputs=decoding_strategy,
        outputs=temperature,
    )
    decoding_strategy.change(
        fn=lambda selection: gr.Slider(visible=(selection in ["Top P Sampling"])),
        inputs=decoding_strategy,
        outputs=top_p,
    )

    gr.ChatInterface(
        fn=model_inference,
        chatbot=chatbot,
        examples=EXAMPLES,
        multimodal=True,
        cache_examples=False,
        additional_inputs=[
            model_selector,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],   
    )

with gr.Blocks() as voice:   
    with gr.Row():
        input = gr.Audio(label="Voice Chat", sources="microphone", type="filepath", waveform_options=False)
        output = gr.Audio(label="OpenGPT 4o", type="filepath",
                        interactive=False,
                        autoplay=True,
                        elem_classes="audio")
        gr.Interface(
            fn=respond, 
            inputs=[input],
                outputs=[output], live=True)
 
with gr.Blocks(theme=theme, css="footer {visibility: hidden}textbox{resize:none}", title="GPT 4o DEMO") as demo:
    gr.Markdown("# OpenGPT 4o")
    gr.TabbedInterface([img, voice], ['💬 SuperChat','🗣️ Voice Chat', ])

demo.queue(max_size=20)
demo.launch()
