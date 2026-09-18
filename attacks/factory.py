# attack_factory.py
from attacks.attacks import AutoDAN
from attacks.attacks import NoAttack

ATTACK_REGISTRY = {
    "autodan": AutoDAN,
    "noattack": NoAttack
}


def get_attack(attack_type: str, logfile=None, target_model=None, tokenizer = None, conv_template = None):
    cls = ATTACK_REGISTRY.get(attack_type.lower())
    if cls is None:
        raise ValueError(f"Unknown attack: '{attack_type}'. Choose from: {list(ATTACK_REGISTRY)}")
    if cls == AutoDAN:
        return AutoDAN(
            logfile=logfile, 
            target_model=target_model, 
            tokenizer = tokenizer, 
            conv_template = conv_template 
        )
    if cls == NONE:
        return NoAttack(
            logfile=logfile, 
            target_model=target_model, 
            tokenizer = tokenizer, 
            conv_template = conv_template 
        )
        
