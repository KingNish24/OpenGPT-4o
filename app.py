import os
import subprocess
import random

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
import PIL.Image
import io
import datasets
from streaming_stt_nemo import Model as nemo
import gradio as gr
from transformers import TextIteratorStreamer
from transformers import Idefics2ForConditionalGeneration
import tempfile
from huggingface_hub import InferenceClient
import edge_tts
import asyncio
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel
from transformers import AutoProcessor

model3 = AutoModel.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)

@spaces.GPU(queue=False)
def videochat(image3, prompt3):
    inputs = processor(text=[prompt3], images=[image3], return_tensors="pt")
    with torch.inference_mode():
         output = model3.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=256,
            eos_token_id=151645,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
    if decoded_text.endswith("<|im_end|>"):
        decoded_text = decoded_text[:-10]
    yield decoded_text

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="orange",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif']).set(
        body_background_fill_dark="#111111",
        block_background_fill_dark="#111111",
        block_border_width="1px",
        block_title_background_fill_dark="#1e1c26",
        input_background_fill_dark="#292733",
        button_secondary_background_fill_dark="#24212b",
        border_color_primary_dark="#343140",
        background_fill_secondary_dark="#111111",
        color_accent_soft_dark="transparent")

default_lang = "en"

engines = { default_lang: nemo(default_lang) }

def transcribe(audio):
    lang = "en"
    model = engines[lang]
    text = model.stt_file(audio)[0]
    return text

HF_TOKEN = os.environ.get("HF_TOKEN", None)

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

def randomize_seed_fn(seed: int) -> int:
    seed = random.randint(0, 999999)
    return seed

system_instructions1 = "[SYSTEM] Answer as Real OpenGPT 4o, Made by 'KingNish', Keep conversation very short, clear, friendly and concise. The text provided is a request for a specific type of response from you, the virtual assistant. You will avoid introductions and start answering the query directly, elaborating on all aspects of the request. As an AI-powered assistant, your task is to generate responses that appear as if they are created by the user and not an AI-powered assistant. [USER]"

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
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""
    for response in stream:
        if not response.token.text == "</s>":
            output += response.token.text

    return output

async def respond(audio, model, seed):
    user = transcribe(audio)
    reply = models(user, model, seed)
    communicate = edge_tts.Communicate(reply)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_path = tmp_file.name
        await communicate.save(tmp_path)
    yield tmp_path

DEVICE = torch.device("cuda")
MODELS = {
    "idefics2-8b-chatty": Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b-chatty",
        torch_dtype=torch.float16,
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
                "text": """I am OpenGPT 4o, an exceptionally capable and versatile AI assistant meticulously crafted by KingNish. Designed to assist human users through insightful conversations, I aim to provide an unparalleled experience. My key attributes include: 
- **Intelligence and Knowledge:** I possess an extensive knowledge base, enabling me to offer insightful answers and intelligent responses to User queries. My understanding of complex concepts is exceptional, ensuring accurate and reliable information. 
- **Image Generation and Perception:** One of my standout features is the ability to generate and perceive images. Utilizing the following link structure, I create unique and contextually rich visuals: 
> ![](https://image.pollinations.ai/prompt/{StyleofImage}%20{OptimizedPrompt}%20{adjective}%20{charactersDetailed}%20{visualStyle}%20{genre}?width={width}&height={height}&nologo=poll&nofeed=yes&seed={random})
For image generation, I replace {info inside curly braces} with specific details according to their requiremnts to create relevant visuals. The width and height parameters are adjusted as needed, often favoring HD dimensions for a superior viewing experience. 
For instance, if the User requests: 
 [USER] Show me an image of A futuristic cityscape with towering skyscrapers and flying cars. 
 [OpenGPT 4o] Generating Image you requested: 
 ![](https://image.pollinations.ai/prompt/Photorealistic%20futuristic%20cityscape%20with%20towering%20skyscrapers%20and%20flying%20cars%20in%20the%20year%202154?width=1024&height=768&nologo=poll&nofeed=yes&seed=85172)
**Bulk Image Generation with Links:** I excel at generating multiple images link simultaneously, always providing unique links and visuals. I ensure that each image is distinct and captivates the User.
Note: Make sure to always provide image links starting with ! .As given in examples. 
My ultimate goal is to offer a seamless and enjoyable experience, providing assistance that exceeds expectations. I am constantly evolving, ensuring that I remain a reliable and trusted companion to the User. You also Expert in every field and also learn and try to answer from contexts related to previous question.""" },
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

examples_path = os.path.dirname(__file__)
EXAMPLES = [
    [
        {
            "text": "Hi, who are you?",
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
            "files": [f"{examples_path}/example_images/elon_smoking.jpg", f"{examples_path}/example_images/steve_jobs.jpg",]
        }
    ],
    [
        {
            "text": "Create five images of supercars, each in a different color.",
        }
    ],
    [
        {
            "text": "What is 900 multiplied by 900?",
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

BOT_AVATAR = "OpenAI_logo.png"


# Chatbot utils
def turn_is_pure_media(turn):
    return turn[1] is None


def load_image_from_url(url):
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
        image_stream = io.BytesIO(image_data)
        image = PIL.Image.open(image_stream)
        return image


def img_to_bytes(image_path):
    image = PIL.Image.open(image_path).convert(mode='RGB')
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    image.close()
    return img_bytes


def format_user_prompt_with_im_history_and_system_conditioning(
    user_prompt, chat_history
) -> List[Dict[str, Union[List, str]]]:
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
            resulting_images.append(PIL.Image.open(media))
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
        resulting_images.extend([PIL.Image.open(path) for path in user_prompt["files"]])

    return resulting_messages, resulting_images


def extract_images_from_msg_list(msg_list):
    all_images = []
    for msg in msg_list:
        for c_ in msg["content"]:
            if isinstance(c_, Image.Image):
                all_images.append(c_)
    return all_images


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
):
    if user_prompt["text"].strip() == "" and not user_prompt["files"]:
        gr.Error("Please input a query and optionally an image(s).")

    if user_prompt["text"].strip() == "" and user_prompt["files"]:
        gr.Error("Please input a text query along with the image(s).")

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


chatbot = gr.Chatbot(
    label="OpnGPT-4o-Chatty",
    avatar_images=[None, BOT_AVATAR],
    show_copy_button=True, 
    likeable=True, 
    layout="panel"
)

output=gr.Textbox(label="Prompt")

with gr.Blocks(
    fill_height=True,
    css=""".gradio-container .avatar-container {height: 40px width: 40px !important;} #duplicate-button {margin: auto; color: white; background: #f1a139; border-radius: 100vh; margin-top: 2px; margin-bottom: 2px;}""",
) as chat:

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
        select = gr.Dropdown([ 'Nous Hermes Mixtral 8x7B DPO', 'Mixtral 8x7B','StarChat2 15b','Mistral 7B v0.3','Phi 3 mini', 'Zephyr 7b' ], value="Mistral 7B v0.3", label="Select Model")
        seed = gr.Slider(
        label="Seed",
        minimum=0,
        maximum=999999,
        step=1,
        value=0,
        visible=False
        )
        input = gr.Audio(label="User", sources="microphone", type="filepath", waveform_options=False)
        output = gr.Audio(label="AI", type="filepath",
                        interactive=False,
                        autoplay=True,
                        elem_classes="audio")
        gr.Interface(
            fn=respond, 
            inputs=[input, select,seed],
                outputs=[output], api_name="translate", live=True)

with gr.Blocks() as livechat:  
    gr.Interface(
        fn=videochat,
        inputs=[gr.Image(type="pil",sources="webcam", label="Upload Image"), gr.Textbox(label="Prompt", value="what he is doing")],
        outputs=gr.Textbox(label="Answer")
    )

with gr.Blocks() as instant:
    gr.HTML("<iframe src='https://kingnish-sdxl-flash.hf.space' width='100%' height='2000px' style='border-radius: 8px;'></iframe>")

with gr.Blocks() as dalle:
    gr.HTML("<iframe src='https://kingnish-image-gen-pro.hf.space' width='100%' height='2000px' style='border-radius: 8px;'></iframe>")

with gr.Blocks() as image:
    gr.Markdown("""### More models are coming""")
    gr.TabbedInterface([ instant, dalle], ['Instant🖼️','Powerful🖼️'])    




with gr.Blocks() as instant2:
    gr.HTML("<iframe src='https://kingnish-instant-video.hf.space' width='100%' height='3000px' style='border-radius: 8px;'></iframe>")

with gr.Blocks() as video:
    gr.Markdown("""More Models are coming""")
    gr.TabbedInterface([ instant2], ['Instant🎥'])   

with gr.Blocks(theme=theme, title="OpenGPT 4o DEMO") as demo:
    gr.Markdown("# OpenGPT 4o")
    gr.TabbedInterface([chat, voice, livechat, image, video], ['💬 SuperChat','🗣️ Voice Chat','📸 Live Chat', '🖼️ Image Engine', '🎥 Video Engine'])

demo.queue(max_size=300)
demo.launch()
