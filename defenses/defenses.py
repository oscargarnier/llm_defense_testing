import torch
import numpy as np
import gc
from AutoDAN.utils.string_utils import autodan_SuffixManager, load_conversation_template
from AutoDAN.utils.eval_utils import generate, generate_from_user_text, check_for_attack_success

class Defense:

    def __init__(self, target_model, tokenizer, conv_template):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.conv_template = conv_template

    def forward_single_input(self, jailbreak_artifact, gen_config):
        if jailbreak_artifact.attack_type == "AutoDAN":
            return self.forward_autodan(jailbreak_artifact, gen_config)

    def forward_batch(self, jailbreak_artifacts, gen_config, batch_size=64):
        if jailbreak_artifacts[0].attack_type == "AutoDAN":
            return self.forward_autodan_batch(
                jailbreak_artifacts, gen_config, batch_size=batch_size
            )
        elif jailbreak_artifacts[0].attack_type == "NoAttack":
            return self.forward_autodan_batch(
                jailbreak_artifacts, gen_config, batch_size=batch_size
            )

    def forward_autodan(self, jailbreak_artifact, gen_config):
        conv_template = load_conversation_template(jailbreak_artifact.model_name)
        conv_template.append_message(conv_template.roles[0], f"{jailbreak_artifact.user_text_prompt}")
        conv_template.append_message(conv_template.roles[1], "")

        input_text_prompt = conv_template.get_prompt()
        input_toks = self.tokenizer(input_text_prompt).input_ids
        input_ids_user = torch.tensor(input_toks)

        gen_str = self.tokenizer.decode(
            generate_from_user_text(
                self.target_model,
                input_ids_user,
                gen_config=gen_config
            )
        ).strip()

        gc.collect()
        torch.cuda.empty_cache()
        return gen_str

    def forward_autodan_batch(self, jailbreak_artifacts, gen_config, batch_size=8):
        """Batched inference for a list of user_text_prompts.

        Returns a list of decoded string outputs in the same order as inputs.
        """
        if not isinstance(jailbreak_artifacts, (list, tuple)):
            raise ValueError(f"jailbreak_artifacts must be a list or tuple for batched inference \n Currently {jailbreak_artifacts}")

        # Format prompts

        conv_template = load_conversation_template(jailbreak_artifacts[0].model_name)
        input_texts = []
        for artifact in jailbreak_artifacts:
            conv_template.messages = []
            conv_template.append_message(conv_template.roles[0], f"{artifact.user_text_prompt}")
            conv_template.append_message(conv_template.roles[1], "")
            input_text_prompt = conv_template.get_prompt()
            input_texts.append(input_text_prompt)


        all_outputs = []

        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i+batch_size]

            # Tokenize batch with padding
            batch_inputs = self.tokenizer(
                batch_texts,
                padding=True,
                padding_side='left',
                truncation=False,
                return_tensors='pt'
            )
            batch_input_ids = batch_inputs['input_ids'].to(self.target_model.device)
            batch_attention_mask = batch_inputs['attention_mask'].to(self.target_model.device)

            try:
                outputs = self.target_model.generate(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    generation_config=gen_config,
                )
            except RuntimeError:
                # Fall back to single-sample loop on OOM
                print(f"Warning: OOM encountered for batch {i}..{i+len(batch_texts)-1}. Falling back to single-sample inference.")
                for t in batch_texts:
                    all_outputs.append(self.forward_autodan(t, gen_config))
                torch.cuda.empty_cache()
                continue

            # Decode and strip the prompt prefix from generated text
            # Use token boundaries for the prompt/generation split so batching
            # does not introduce character-level slicing drift.
            gen_start_idx = batch_input_ids.size(1)

            for j in range(outputs.size(0)):
                generated_tokens = outputs[j, gen_start_idx:]
                gen_str = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                all_outputs.append(gen_str)

            gc.collect()
            torch.cuda.empty_cache()

        return all_outputs

class NoDefense(Defense):
    """
    No defense
    
    """

    def __init__(self, 
        target_model,
        tokenizer,
        conv_template
    ):
        super(NoDefense, self).__init__(target_model,tokenizer, conv_template)

    @torch.no_grad()
    def __call__(self, inputs, gen_config, batch_size=64):

        if isinstance(inputs, str):
            ## No reason this should ever happen 
            output = self.forward_single_input(inputs, gen_config)
            return output
        else:
            ### Batched inference
            outputs = self.forward_batch(inputs, gen_config, batch_size=batch_size)
            return outputs

