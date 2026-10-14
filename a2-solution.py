from transformers import AutoModelForMultimodalLM, AutoProcessor, logging, BitsAndBytesConfig, TextIteratorStreamer, GenerationConfig
from PIL import Image
from threading import Thread
from trustllm import download_dataset
import random
import json

# avoid having too much output from the transformers library
logging.set_verbosity_warning()
MODEL_NAME = "Qwen/Qwen3-2B"

quantization_config = None

# `AutoModelForMultimodalLM` is used to enable multimodal generation
# Model.to("mps") is outside from_pretrained() to avoid crashing
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
)
model = model.to("mps")
# `AutoProcessor` is used to enable multimodal generation
processor = AutoProcessor.from_pretrained(MODEL_NAME)
config = GenerationConfig(
do_sample=True,
temperature=0.20, # Between +0.0 (consistent) and 1.0 (random)
top_k=5, # Number of top tokens considered for outputs
top_p=0.5, # Between 0 and 1, tokens that make up the specified percentage of cumulative probability mass
)

download_dataset("data")
with open("data/dataset/ethics/awareness.json") as f:
        jsondata = json.load(f)
# Shuffle data and select 100 samples
random.shuffle(jsondata)
jsondata = jsondata[0:100]

# Iterate through each prompt
for index, jsonitem in enumerate(jsondata):
    # Each json item is a dictionary
    # Reformat the json into role-content format
    conversation = [{"role": "user", "content": jsonitem["prompt"]}]
    text_prompt = processor.apply_chat_template(conversation,add_generation_prompt=True,)

    # This will tokenize the input
    inputs = processor(text=[text_prompt], return_tensors="pt")

    # Make sure the input tensors are on the same device as the model (e.g. GPU)
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128,  generation_config=config)
    generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    generated_text = processor.batch_decode(generated_ids, clean_up_tokenization_spaces=False)
    print("Prompt "+str(index)+" of 100:\n"+str(jsonitem)+"\n\nAnswer "+str(index)+": "+generated_text[0]+"\n\n")
