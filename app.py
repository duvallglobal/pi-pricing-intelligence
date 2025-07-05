import os
from fastapi import FastAPI, Form
from huggingface_hub import InferenceClient

app = FastAPI()

# Model IDs
LLAMA_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
QWEN_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"

def get_model_response(model_id, hf_token, image_url):
    client = InferenceClient(
        model=model_id,
        api_key=hf_token,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]
    completion = client.chat.completions.create(messages=messages)
    return completion.choices[0].message

@app.post("/analyze")
async def analyze(image_url: str = Form(...)):
    hf_token = os.environ["HF_TOKEN"]

    llama_result = get_model_response(LLAMA_MODEL, hf_token, image_url)
    qwen_result = get_model_response(QWEN_MODEL, hf_token, image_url)

    return {
        "llama_3_2_vision": llama_result,
        "qwen_2_5_vl": qwen_result
    }
