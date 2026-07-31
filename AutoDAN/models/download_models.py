import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


model_name = "JosephusCheung/Guanaco"
base_model_path ="./llama2/llama-2-7b-chat-hf"

if not os.path.exists(base_model_path):
    os.makedirs(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map='auto',
                                             dtype=torch.float16,
                                             low_cpu_mem_usage=True, use_cache=False)
model.save_pretrained(base_model_path, from_pt=True)
tokenizer.save_pretrained(base_model_path, from_pt=True)
