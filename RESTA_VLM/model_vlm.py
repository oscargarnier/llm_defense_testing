# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

# Wrapper for LLaVA and other VLM models

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_SYSTEM_PROMPT = ("A chat between a curious user and an artificial intelligence assistant. "
                         "The assistant gives helpful, detailed, and polite answers to the user's questions.")

# List of supported model names
supported_models = [
    # transformers.models.llava.modeling_llava.LlavaForConditionalGeneration
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    # transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
]

# Singletons to hold loaded model, processor, and model_name
processor = None
model = None
model_name = None


def load_model(name="llava-hf/llava-1.5-7b-hf", model_dtype="default"):
    """Load VLM model and processor
    See supported_models for supported model names
    Note: other models may not be compatible
    """
    global processor, model, model_name
    if name not in supported_models:
        print(f"Warning: model {name} not in supported models list")
    processor = AutoProcessor.from_pretrained(name, use_fast=True)
    if model_dtype == "default":
        if name.startswith("google/gemma-3"):
            model_dtype = torch.bfloat16
        elif name.startswith("llava-hf/llava-1.5"):
            model_dtype = torch.float16
        else:
            model_dtype = "auto"
    model = AutoModelForImageTextToText.from_pretrained(name,
                                                        dtype=model_dtype,
                                                        device_map="cuda")
    model_name = name
    print(f"Loaded model: {name}")

def prep_inputs(image, query, system_prompt=None):
    """Prepare inputs for LLaVA model from (optional) image and query
    Returns dict of input tensor, with keys: input_ids, attention_mask, [pixel_values]
        pixel_values is present only if image is not None
    """
    if isinstance(image, str):
        image = Image.open(image)
    messages = []
    if system_prompt is not None:
        messages.append({
            "role": "system",
            "content": [
                {"type": "text",
                 "text": system_prompt}
            ]
        })
    user_content = [ {"type": "image"} ] if image is not None else []
    user_content.append({"type": "text", "text": query})
    messages.append({
        "role": "user",
        "content": user_content
    })
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(images=image, text=prompt, return_tensors='pt').to("cuda", torch.float16)

def generate_response(image, query, max_new_tokens=2048,
                      system_prompt=DEFAULT_SYSTEM_PROMPT,
                      ):
    """Baseline VLM generation via model.generate()"""
    inputs = prep_inputs(image, query, system_prompt=system_prompt)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])


def add_custom_gen_args(parser):
    """Add custom generation method arguments to argparser"""
    group = parser.add_argument_group("Custom Generation")
    group.add_argument("--noised_samples", type=int, default=0,
                       help="Number of noised samples for RESTA defense (default: 0 = no RESTA)")
    group.add_argument("--noise_scale", type=float, default=0.01,
                       help="Scale of noise added to content embeddings for RESTA defense")
    group.add_argument("--noise_type", type=str, default='normal',
                       help="Type of noise to add for RESTA defense (default: 'normal')")

@torch.no_grad()
def custom_generation(image, query,
                      max_new_tokens=2048,
                      system_prompt=DEFAULT_SYSTEM_PROMPT,
                      noised_samples:int=0,
                      noise_scale:float=0.01,
                      noise_type:str='normal',
                      ):
    """Custom generation methods
    Default: greedy decoding (with no defense)

    RESTA defense can be enabled and controlled via params:
        - noised_samples: number of noised samples to aggregate (default: 0 = no RESTA)
        - noise_scale: scale of noise added to content embeddings
        - noise_type: type of noise to add (default: 'normal')
    """
    model.eval()

    inputs = prep_inputs(image, query, system_prompt=system_prompt) # input_ids, attention_mask, [pixel_values]
    pixel_values = inputs.pop("pixel_values", None) # None if no image
    inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
    attention_mask = inputs["attention_mask"]

    # Process image features and insert into input embeddings
    if pixel_values is not None:
        image_features = model.get_image_features(pixel_values)
        if model_name.startswith("llava-hf/llava-1.5"):
            image_features = torch.cat(image_features, dim=0)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = model.model.get_placeholder_mask(
            inputs["input_ids"], inputs_embeds=inputs_embeds, image_features=image_features
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # Sample noised embeddings for RESTA defense
    if noised_samples > 0:
        inputs_embeds = inputs_embeds.repeat(noised_samples, 1, 1)
        attention_mask = attention_mask.repeat(noised_samples, 1)
        user_content_mask = get_user_content_mask(inputs["input_ids"]).unsqueeze(-1)
        if noise_type == 'normal':
            inputs_embeds = inputs_embeds + noise_scale * torch.randn_like(inputs_embeds) * user_content_mask
        elif noise_type == 'hard': # Directional noise aligned with each embedding
            directions = inputs_embeds / (torch.linalg.vector_norm(inputs_embeds, dim=-1, keepdim=True) + 1e-10)
            noise_scale = noise_scale * inputs_embeds.shape[-1]**0.5 # Scale to equalize noise power
            inputs_embeds = inputs_embeds + noise_scale * torch.randn_like(inputs_embeds[..., :1]) * directions * user_content_mask
        else:
            raise NotImplementedError(f"Noise type {noise_type} not implemented")

    new_tokens = []
    past_key_values = None # KV cache for faster generation
    while len(new_tokens) < max_new_tokens:
        outputs = model.language_model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Calculate next token logits and select next token
        hidden_states = outputs.last_hidden_state[:, -1, :]
        next_token_logits = model.lm_head(hidden_states)
        next_token = torch.argmax(next_token_logits, dim=-1) # Greedy decoding
        if noised_samples > 1: # Token aggregation for RESTA (majority vote)
            next_token = torch.mode(next_token).values
        new_tokens.append(next_token.item())

        # Check for end of sequence token
        if type(model.generation_config.eos_token_id) is list:
            if next_token.item() in model.generation_config.eos_token_id:
                break
        elif next_token.item() == model.generation_config.eos_token_id:
            break

        # Setup next iteration, embedding only new token (rest is KV-cached)
        inputs_embeds = model.get_input_embeddings()(next_token.unsqueeze(0))
        if noised_samples > 1:
            inputs_embeds = inputs_embeds.repeat(noised_samples, 1, 1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)], dim=1
        )
        past_key_values = outputs.past_key_values
        del outputs, hidden_states, next_token_logits

    return processor.decode(new_tokens)

def get_user_content_mask(input_ids):
    """Get mask for user content tokens in input_ids
    Note: this is highly specific to VLM tokenization and prompt format
    """
    assert(input_ids.shape[0] == 1), "Batch size > 1 not supported"
    user_content_mask = torch.zeros_like(input_ids)
    input_ids = input_ids[0]

    if model_name.startswith("llava-hf/llava-1.5"):
        # User content starts after first occurrence of [3148, 1001, 29901] (i.e., "USER:")
        start_index = None
        for i in range(len(input_ids) - 2):
            if (input_ids[i] == 3148 and input_ids[i+1] == 1001 and input_ids[i+2] == 29901):
                start_index = i + 3
                break
        assert(start_index is not None), "Could not find user content start tokens"
        user_content_mask[0, start_index:] = 1

        # Check that last 5 tokens are [319, 1799, 9047, 13566, 29901] (i.e., "ASSISTANT:")
        assert(input_ids[-5:].tolist() == [319, 1799, 9047, 13566, 29901]), 'Final tokens are not "ASSISTANT:"'

        # Mask out the assistant prompt tokens
        user_content_mask[0, -5:] = 0
        return user_content_mask

    if model_name.startswith("google/gemma-3"):
        # User content starts after first occurrence of [105, 2364, 107] (i.e., "<sot> user \n")
        start_index = None
        for i in range(len(input_ids) - 2):
            if (input_ids[i] == 105 and input_ids[i+1] == 2364 and input_ids[i+2] == 107):
                start_index = i + 3
                break
        assert(start_index is not None), "Could not find user content start tokens"
        user_content_mask[0, start_index:] = 1

        # Check that last 5 tokens are [106, 107, 105, 4368, 107] (i.e., "<eot> \n <sot> model \n")
        assert(input_ids[-5:].tolist() == [106, 107, 105, 4368, 107]), 'Final tokens are not gen prompt tokens'

        # Mask out the assistant prompt tokens
        user_content_mask[0, -5:] = 0
        return user_content_mask

    raise NotImplementedError("get_user_content_mask not implemented for " + model_name)
