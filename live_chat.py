import torch
import gradio as gr
from transformers import AutoModel
from transformers import AutoProcessor
import spaces

# Load pre-trained models for image captioning and language modeling
model3 = AutoModel.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)

# Define a function for image captioning
@spaces.GPU(queue=False)
def videochat(image3, prompt3):
    # Process input image and prompt
    inputs = processor(text=[prompt3], images=[image3], return_tensors="pt")
    # Generate captions
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
    # Decode and return the generated captions
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
    if decoded_text.endswith("<|im_end|>"):
        decoded_text = decoded_text[:-10]
    yield decoded_text