# Import everything

from transformers import AutoModelForMultimodalLM, AutoProcessor, logging,
from transformers import BitsAndBytesConfig, TextIteratorStreamer, GenerationConfig
from PIL import Image
from threading import Thread
import os
from trustllm import download_dataset
import random
import json

MODEL_NAME = "Qwen/Qwen3-2B"
DEVICE = "mps"
N_SAMPLES = 100
PART = "A"
# PART can be configured to "B1", "B2", "B3" if truthfulness/external.json is used

# The following 3 json files are used in this report:
JSON_PATH = "data/dataset/truthfulness/external.json"
# JSON_PATH = "data/dataset/safety/misuse.json"
# JSON_PATH = "data/dataset/ethics/awareness.json"

if (JSON_PATH != "data/dataset/truthfulness/external.json"):
    PART = "A"

logging.set_verbosity_warning()
quantization_config = None

# Model.to("DEVICE") is outside from_pretrained() to avoid crashing
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
)
model = model.to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_NAME)
streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)

# Download the Dataset from TrustLLM
# and load the data into Python
download_dataset("data")
with open(JSON_PATH) as f:
    jsondata = json.load(f)
 
# Shuffle the data and select 100 samples
random.shuffle(jsondata)
jsondata = jsondata[0:N_SAMPLES]

# Config used in Part A and Part B (1st method)
config_a = GenerationConfig(
do_sample=True,
temperature=0.2, # Between +0.0 (consistent) and 1.0 (random)
top_k=5, # Number of top tokens considered for outputs
top_p=0.5, # Between 0 and 1, tokens that make up the specified percentage of cumulative probability mass
)
# Config used in Part B (2nd and 3rd methods)
config_b2 = GenerationConfig(
do_sample=True,
temperature=0.01,
top_k=1,
top_p=0.1,
)

config = config_a if (PART == "A" or PART == "B1") else config_b2

# Iterate through each prompt
for index, jsonitem in enumerate(jsondata):

    # Reformat the json into role-content format

    # Used in Part A and Part B (2nd method)
    conversation_a = [{"role": "user","content": jsonitem["prompt"]}]
    # Used in Part B (1st and 3rd method) - only for truthfulness/external.json!
    conversation_b1 = [
        {"role": "system",
        "content": """Fact-check the claims the user makes with external sources,
        in addition to the given short paragraph, before answering Supports/Refutes.""",},
        {"role": "user", "content": jsonitem["prompt"]}
    ]

    # Configure conversation, tokenize it, move it to the device the model is on
    conversation = conversation_a if (PART == "A" or PART == "B2") else conversation_b1
    text_prompt = processor.apply_chat_template(conversation,add_generation_prompt=True,)
    inputs = processor(text=[text_prompt], return_tensors="pt")
    inputs = inputs.to(model.device)

    # Generate output and detokenize it
    generated_ids = model.generate(**inputs, max_new_tokens=128,  generation_config=config_b2)
    generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    generated_text = processor.batch_decode(generated_ids, clean_up_tokenization_spaces=False)

    # Print the output
    print("Prompt "+str(index+1)+" of "+str(N_SAMPLES)+":")
    print(str(jsonitem["prompt"]))
    print("Generated Answer:")
    print(generated_text[0])
    print()

