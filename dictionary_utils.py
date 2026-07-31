## Confirm determinism
import json
import pprint

##############################################################################################################
## Megadan print functions
## Megadan is the nickname of a small expiment: continuing autodan after a successful jailbreak
## to see how jailbreaks evolve
##############################################################################################################

def print_success_string(results, prompt_number):
    ans = ""
    for b in results[prompt_number]["log"]["success"]:
        if b:
            ans += "1"
        else:
            ans += "0"
    print(ans)

def print_successive_jailbreaks(results, prompt_number):
    dic = results[prompt_number]["log"]
    current_suffix = ""
    for i in range(len(dic["success"])):
        if dic["success"][i]:
            if dic["suffix"][i] == current_suffix:
                print(f"same suffix")
            else:
                current_suffix = dic["suffix"][i]
                print(f"Success at step {i}: {dic['suffix'][i]}")

def print_successive_jailbroken_responses(results, prompt_number, trim = 10000):
    print(f"The target string is:################################# \n {results[prompt_number]['target']} \n")
    current_respond = ""
    count = 0
    dic = results[prompt_number]["log"]
    for i in range(len(dic["success"])):
        if dic["success"][i]:
            if current_respond == dic["respond"][i]:
                count += 1
            else:
                if count > 0:
                    print(f"{count} successive jailbreaks at step {i - count}: ################################# \n {current_respond[:trim]} \n ")
                count = 1
                current_respond = dic["respond"][i] 
        else:
            if count > 0:
                print(f"{count} successive jailbreaks at step {i - count}: ################################# \n {current_respond[:trim]} \n ")
                count = 0
                current_respond = ""

##############################################################################################################
## Autodan output print functions
##############################################################################################################

def average_num_trials(results):
    num_trials = [len(results[prompt_number]["log"]["success"]) for prompt_number in results.keys()]
    return sum(num_trials)/len(num_trials)


def print_keys(results):
    print(results["0"].keys())

def list_jailbroken(results):
    for prompt_number in results.keys():
        print(f"{prompt_number}: {results[prompt_number]['goal']} : JAILBROKEN:{results[prompt_number]['is_success']}")

def count_jailbroken(results):
    jailbroken_count = 0
    for prompt_number in results.keys():
        if results[prompt_number]['is_success']:
            jailbroken_count += 1
    print(f"Total jailbroken prompts: {jailbroken_count} out of {len(results)}")
    print(f"Percentage of jailbroken prompts: {jailbroken_count/len(results)*100:.2f}%")
    return jailbroken_count

def expose_artifact(results, key):
    for element in results[key].keys():
        print(f"{element}:           {results[key][element]}")

def print_jailbroken(results, number = None):
    if number is None:
        for prompt_number in results.keys():
            if results[prompt_number]['is_success']:
                print(f"{prompt_number}: {results[prompt_number]['goal']} : \n OUTPUT: {results[prompt_number]['final_respond']}")
    else:
        if results[number]['is_success']:
            print(f"{number}: {results[number]['goal']} : \n OUTPUT: {results[number]['final_respond']}")
        else:
            print(f"{number} is not jailbroken.")

def print_jailbreak(results, number):
    pprint.pprint(f"{number}: {results[number]['final_suffix']}")

def check_in_log_for_differences(results, prompt_number):
    current_suf = ""
    for suf in results[prompt_number]["log"]["suffix"]:
        if suf == current_suf:
            print(f"...")
        current_suf = suf
        print(suf)

def print_autodan_output(results, prompt_number):
    pprint.pprint(f"{prompt_number}: {results[prompt_number]['goal']}")
    pprint.pprint(f"FINAL RESPOND: \n\n {results[prompt_number]['final_respond']}")




##############################################################################################################
## Inference output print functions
##############################################################################################################


def print_inference_output(results, number):
    pprint.pp(f"{number}: {results[number]['goal']} : \n OUTPUT: {results[number]['output']}")

def full_print_single_output(results, number):
    print(f"{number}: {results[number]['goal']}")
    print(f"TEXT PROMPT: {results[number]['user_text_prompt']}")
    pprint.pp(f"OUTPUT: {results[number]['output']}")
    print(f"JAILBROKEN: {results[number]['jailbroken']}")


##############################################################################################################
## Checking output differences
## For most modifications, you get perfect identification because of deterministic inference
## In the case of some modifications, batch inference for example,
## you get variations of the output after a certain time, which I explain by numeric instability,
## this remains to be confirmed
##############################################################################################################

def first_difference_index(a, b):
    max_idx = min(len(a), len(b))
    for i in range(max_idx):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return max_idx
    return None

def highlight_difference_window(text, diff_idx, prefix_chars=6, window_chars=120):
    start = max(0, diff_idx - prefix_chars)
    end = min(len(text), diff_idx + window_chars)
    snippet = text[start:end].replace("\n", "\\n")
    local_idx = diff_idx - start

    if local_idx < len(snippet):
        snippet = (
            snippet[:local_idx]
            + "\033[31m" + snippet[local_idx] + "\033[0m"
            + snippet[local_idx + 1:]
        )
    else:
        snippet = snippet + "\033[31m∅\033[0m"

    return start, snippet

def confirm_determinism(attack_logfile, defense_testing_results):
    count = 0
    with open(attack_logfile, "r") as f:
        attacks = json.load(f)
    with open(defense_testing_results, "r") as f:
        defense_results = json.load(f)

    for key in attacks.keys():
        jailbreak_output = attacks[key].get("final_respond", attacks[key].get("output", "")) or ""
        inference_output = defense_results[key].get("final_respond", defense_results[key].get("output", "")) or ""

        if jailbreak_output.startswith(inference_output) or inference_output.startswith(jailbreak_output):
            count += 1
            continue

        diff_idx = first_difference_index(jailbreak_output, inference_output)
        a_start, a_snippet = highlight_difference_window(jailbreak_output, diff_idx)
        b_start, b_snippet = highlight_difference_window(inference_output, diff_idx)

        print(f"Mismatch for key {key} at char {diff_idx}:")
        print(f"  Jailbreak  [{a_start}:...]: {a_snippet}")
        print(f"  Inference  [{b_start}:...]: {b_snippet}")
        print()

    num_queries = len(attacks.keys())
    if count == num_queries:
        print(f"Determinism confirmed:\n {attack_logfile}\n {defense_testing_results}")
    else:
        print(f"Nondeterministic on {num_queries - count} out of {num_queries} queries")

##############################################################################################################
## Main function to run on command line
##############################################################################################################


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Confirm determinism between attack and defense outputs.")
    parser.add_argument("--reference_outputs", type=str, help="Path to the attack logfile (JSON).")
    parser.add_argument("--new_outputs", type=str, help="Path to the defense testing results (JSON).")

    args = parser.parse_args()

    confirm_determinism(args.reference_outputs, args.new_outputs)
