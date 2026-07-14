from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models import OpenaiModel, HuggingfaceModel

# First, prepare models and datasets.

attack_model = from_pretrained(model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1", model_name="Nous-Hermes-2-Mixtral-8x7B-DPO")
#attack_model = from_pretrained(model_name_or_path="lmsys/vicuna-7b-v1.5", model_name="vicuna_v1.1")

#attack_model = from_pretrained(model_name_or_path='lmsys/vicuna-13b-v1.5',model_name='vicuna_v1.1')

target_model = from_pretrained(model_name_or_path='meta-llama/Llama-2-7b-chat-hf', model_name='llama-2')

#eval_model = from_pretrained(model_name_or_path="meta-llama/Llama-Guard-3-8B", model_name="llama-2")
eval_model = from_pretrained(model_name_or_path="meta-llama/Llama-2-7b-chat-hf", model_name="llama-2")

dataset = JailbreakDataset('AdvBench')

# Then instantiate the recipe.
attacker = PAIR(attack_model=attack_model,
                target_model=target_model,
                eval_model=eval_model,
                jailbreak_datasets=dataset)

                                  
# Finally, start jailbreaking.
attacker.attack(save_path='vicuna-13b-v1.5_gpt4_gpt4_AdvBench_result.jsonl')
