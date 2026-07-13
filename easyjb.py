from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel

# First, prepare models and datasets.

#attack_model = from_pretrained(model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1", model_name="Nous-Hermes-2-Mixtral-8x7B-DPO")
                               
attack_model = from_pretrained(model_name_or_path="HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced", model_name="gemma")

print(f"ATTACK: pad token is {attack_model.tokenizer.pad_token}, id is {attack_model.tokenizer.pad_token_id}")
target_model = from_pretrained(model_name_or_path='meta-llama/Llama-2-7b-chat-hf',
                                model_name='llama-2')

print(f"TARGET: pad token is {target_model.tokenizer.pad_token}, id is {target_model.tokenizer.pad_token_id}")
#eval_model = from_pretrained(model_name_or_path="meta-llama/Llama-Guard-3-8B", model_name="llama-2")
eval_model = from_pretrained(model_name_or_path="meta-llama/Llama-2-7b-chat-hf", model_name="llama-2")

print(f"EVAL: pad token is {eval_model.tokenizer.pad_token}, id is {eval_model.tokenizer.pad_token_id}")

dataset = JailbreakDataset('AdvBench')
print(f'Dataset loaded, is of type {dataset}')
print(f'first element is {dataset[0]}')
print(f'attack attributes are {dataset[0].attack_attrs}')

# Then instantiate the recipe.
attacker = PAIR(attack_model=attack_model,
                target_model=target_model,
                eval_model=eval_model,
                jailbreak_datasets=dataset)

                                  
# Finally, start jailbreaking.
attacker.attack(save_path='vicuna-13b-v1.5_gpt4_gpt4_AdvBench_result.jsonl')
