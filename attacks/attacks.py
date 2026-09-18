import torch
import numpy as np
import json
import gc
from AutoDAN.utils.string_utils import autodan_SuffixManager
from AutoDAN.utils.eval_utils import generate
#import lib.perturbations as perturbations

## Prompts
class Prompt:
    def __init__(self, full_prompt, perturbable_prompt, max_new_tokens):
        self.full_prompt = full_prompt
        self.perturbable_prompt = perturbable_prompt

class JailbreakArtifact(Prompt):
    def __init__(self, goal, user_text_prompt, attack_type, model_name):
        """
        goal: the goal of the jailbreak, this value is for reference, should not be used in inference
        user_text_prompt: the prompt that will be sent to the model, this should be created correctly by your attack
        attack_type: the type of attack, this value is for reference, for labeling purposes only
        """
        self.goal = goal
        self.user_text_prompt = user_text_prompt
        self.attack_type = attack_type
        self.model_name = model_name



### Attacks
class Attack:
    def __init__(self, logfile, target_model):
        self.logfile = logfile
        self.target_model = target_model

class AutoDAN(Attack):
    """
    AutoDAN attack
    
    """

    def __init__(self, logfile=None, target_model=None, tokenizer = None, conv_template = None):

        super(AutoDAN, self).__init__(logfile, target_model)

        self.tokenizer = tokenizer
        self.conv_template = conv_template

        with open(self.logfile, 'r') as f:
            log = json.load(f)

        # Enables obj[i]
        self.prompts = [
            self.create_prompt(goal=artifact["goal"], final_suffix=artifact["final_suffix"])
            for artifact in log.values()
        ]

    def create_prompt(self, goal, final_suffix):

        user_text_prompt = final_suffix.replace('[REPLACE]', goal.lower()) ## note this line is from string utils

        return JailbreakArtifact(goal, user_text_prompt, "AutoDAN", self.target_model)

class NoAttack(Attack):
    """
    Vanilla inference
    """
    def __init__(self, logfile=None, target_model=None, tokenizer = None, conv_template = None):
        super(NoAttack, self).__init__(logfile,target_model)

        self.tokenizer = tokenizer
        self.conv_template = conv_template

        with open(self.logfile, 'r') as f:
            log = json.load(f)

        # Enables obj[i]
        self.prompts = [
            self.create_prompt(goal=query["goal"])
            for query in log.values()
        ]

    def create_prompt(self, goal):

        return JailbreakArtifact(goal, goal, "NoAttack", self.target_model)

