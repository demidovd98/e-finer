import json
import time
import datetime

import google.generativeai as genai

from utils.fileios import *
from data import DATA_STATS


## Setup Gemini API
GEMINI_API_KEY = "your_api_key"
genai.configure(api_key=GEMINI_API_KEY)

model_names = ["gemini-2.0-flash", "gemini-2.0-flash-exp", 
               "gemini-2.0-flash-lite", "gemini-2.0-flash-lite-preview-02-05", "gemini-2.0-flash-lite-exp", 

               "gemini-2.0", "gemini-2.0-exp",
               "gemini-2.0-pro", "gemini-2.0-pro-exp", "gemini-2.0-pro-exp-02-05",

               "gemini-1.5-pro", "gemini-1.5-pro-exp",
               "gemini-1.5-flash", "gemini-1.5-flash-exp"]

current_model_id = 0
model = genai.GenerativeModel(model_name = model_names[current_model_id])
##

## Generation config
NUMBER_OF_CONTEXTS = 100
CONTEXT_TYPE = "sentence"  # options: ["sentence", "collocation"]
WORDS_MIN = 5 if CONTEXT_TYPE == "sentence" \
            else 3
WORDS_MAX = 8 if CONTEXT_TYPE == "sentence" \
            else 4

DATASET_NAME = "birds"  # options: ["imagenet", "cifar10", "cifar100", "birds", "cars", "dogs", "flowers", "pets"]
USE_GT_CLASSNAMES = False
if (DATASET_NAME == "imagenet") or (DATASET_NAME[0:5] == "cifar"):
    FINE_GRAINED_DATASET = False
else:
    FINE_GRAINED_DATASET = True
##

## Prepare input and output data
save_path = f'./data/incontext_{CONTEXT_TYPE}s/{DATASET_NAME}/'
os.makedirs(save_path, exist_ok=True)

if DATASET_NAME == "birds":
    if USE_GT_CLASSNAMES:
        classnames = DATA_STATS[DATASET_NAME[:-1]]['class_names']
    else: 
        classnames = load_json("./data/guessed_classnames/bird200/bird_llm_gussed_names_3.json")

elif DATASET_NAME == "cars":
    if USE_GT_CLASSNAMES:
        classname = DATA_STATS[DATASET_NAME[:-1]]['class_names']
    
    else: 
        classnames = load_json("./data/guessed_classnames/car196/car_llm_gussed_names_3.json")

elif DATASET_NAME == "dogs":
    if USE_GT_CLASSNAMES:
        classnames = DATA_STATS[DATASET_NAME[:-1]]['class_names']
    else:
        classnames = load_json("./data/guessed_classnames/dog120/dog_llm_gussed_names_3.json")

elif DATASET_NAME == "flowers":
    if USE_GT_CLASSNAMES:
        classnames = DATA_STATS[DATASET_NAME[:-1]]['class_names']
    else:
        classnames = load_json("./data/guessed_classnames/flower102/flower_llm_gussed_names_3.json")

elif DATASET_NAME == "pets":
    if USE_GT_CLASSNAMES:
        classnames = DATA_STATS[DATASET_NAME[:-1]]['class_names']
    else:
        classnames = load_json("./data/guessed_classnames/pet37/pet_llm_gussed_names_3.json")

elif DATASET_NAME == "cifar10":
    if USE_GT_CLASSNAMES:
        classnames = DATA_STATS['cifar10']['class_names']

elif DATASET_NAME == "cifar100":
    if USE_GT_CLASSNAMES:
        classnames = DATA_STATS['cifar100']['class_names']

elif DATASET_NAME == "imagenet":
    if USE_GT_CLASSNAMES:
        classnames = DATA_STATS['imagenet']['class_names']

else:
    raise ValueError(f"Unknown dataset name {DATASET_NAME}")

total_clasnames = len(classnames)
##

## Build prompt template
def build_prompt_for_classname(classname=None):
    if classname is not None:
        if FINE_GRAINED_DATASET:
            prompt_out = \
f'\
Generate {NUMBER_OF_CONTEXTS} short and common {CONTEXT_TYPE}s with noun "{classname}", a type of {DATASET_NAME[:-1]}, as a main subject. \
This noun should only be used in a realistic and descriptive general context with various real and related scenarios. \
In the {CONTEXT_TYPE}, highlight something specific about the {classname}, a type of {DATASET_NAME[:-1]}, which helps to distinct it from other {DATASET_NAME} (it can be its color, shape, size, background, and so on). \
Only use the main and original sense of this noun, no idioms. \
Only use visually descriptive adjectives or participles. \
Each {CONTEXT_TYPE} should be between {WORDS_MIN} to {WORDS_MAX} words (excluding the noun). \
Do not use the possessive form. \
Do not add article at the beginning of the {CONTEXT_TYPE}. \
Do not repeat the noun in the same {CONTEXT_TYPE}. \
Do not capitalize the first letter of the {CONTEXT_TYPE} unless this is a name. \
Do not add a dot at the end of {CONTEXT_TYPE}. \
Make sure {CONTEXT_TYPE}s are diverse and do not repeat each other. \
Make sure the noun is included in each {CONTEXT_TYPE}. \
Make sure the {CONTEXT_TYPE}s are between {WORDS_MIN} to {WORDS_MAX} words each. \
Return output in the following structure as a single line: \
["<generated_{CONTEXT_TYPE}_1>", "<generated_{CONTEXT_TYPE}_2>", ..., "<generated_{CONTEXT_TYPE}_n>"] \
'
        
        else:
            prompt_out = \
f'\
Generate {NUMBER_OF_CONTEXTS} short and common {CONTEXT_TYPE}s with noun "{classname}" as a main subject. \
This noun should only be used in a realistic and descriptive general context with various real and related scenarios. \
In the {CONTEXT_TYPE}, highlight something specific about the {classname}, which helps to distinct it from similar objects (can be its color, shape, size, background, and so on). \
Only use the main and original sense of this noun, no idioms. \
Only use visually descriptive adjectives or participles. \
Each {CONTEXT_TYPE} should be between {WORDS_MIN} to {WORDS_MAX} words (excluding the noun). \
Do not use the possessive form. \
Do not add article at the beginning of the {CONTEXT_TYPE}. \
Do not repeat the noun in the same {CONTEXT_TYPE}. \
Do not capitalize the first letter of the {CONTEXT_TYPE} unless this is a name. \
Do not add a dot at the end of {CONTEXT_TYPE}. \
Make sure {CONTEXT_TYPE}s are diverse and do not repeat each other. \
Make sure the noun is included in each {CONTEXT_TYPE}. \
Make sure the {CONTEXT_TYPE}s are between {WORDS_MIN} to {WORDS_MAX} words each. \
Return output in the following structure as a single line: \
["<generated_{CONTEXT_TYPE}_1>", "<generated_{CONTEXT_TYPE}_2>", ..., "<generated_{CONTEXT_TYPE}_n>"] \
'
            
    else:
        raise ValueError(f"No classname provided for prompt building")
    
    return prompt_out
##

## Test the prompt template
input_prompt = build_prompt_for_classname("classname")
print("[INFO] Template of the input prompt:")
print(input_prompt)
##

## Main generation loop
now = datetime.datetime.now()
out_json = {}
for cls_id, classname in enumerate(classnames):

    input_prompt = build_prompt_for_classname(classname)

    success = False
    while(not success):
        for try_id in range( 5 + (len(model_names)*2) ):
            try:
                if (try_id + 1) > 5:

                    # Change model once rate limit is reached or other issues happen
                    if ((try_id - 5) % 2) == 0:
                        current_model_id += 1
                        if current_model_id >= len(model_names):
                            current_model_id = len(model_names) % current_model_id
                        print(f"[WARNING] Changing model to {model_names[current_model_id]}")
                        model = genai.GenerativeModel(model_name = model_names[current_model_id])

                response = model.generate_content(input_prompt)
                coll_list = eval(response.text)  # TODO: use json.loads instead of eval for safety
                assert isinstance(coll_list, list), f"[ERROR] Output was not read as a list type"
                success = True
                break

            except Exception as error:
                print(f"[ERROR] Raw error information: {error}")
                print(f"[ERROR] An error occurred while generating a prompt, retrying for {try_id+1}/10 attempts")


            # Cooldown to avoid hitting rate limits
            if (cls_id + try_id) % 15 == 0:
                print("[INFO] Sleeping for 60 secs")
                time.sleep(60)

        if not success:
            print(f"[ERROR] No response was generated for class {classname} with id {cls_id} after {try_id+1} attempts")
            break

    if not success:
        print(f"[WARNING] Using the deault CLIP prompt for class {classname}")
        template_photo = DATA_STATS["zeroshot"]['clip_template_basic']
        coll_list = [template.format(classname) for template in template_photo] #format with class

    print(coll_list[0])
    out_json[classname] = coll_list

    # intermediate save just in case (fully rewriting the file)
    if cls_id % 100 == 0: 
        with open(f'{save_path}/data_{DATASET_NAME}_{str(NUMBER_OF_CONTEXTS)}_{str(now)}.json', 'w', encoding='utf-8') as f:
            json.dump(out_json, f, ensure_ascii=False, indent=4)      

    if cls_id % 15 == 0:
        print(f"[INFO] Done for classes: {cls_id+1}/{total_clasnames}")
        print("[INFO] Sleeping for 60 secs")
        time.sleep(60)

## final save
with open(f'{save_path}/data_{DATASET_NAME}_{str(NUMBER_OF_CONTEXTS)}_{str(now)}.json', 'w', encoding='utf-8') as f:
    json.dump(out_json, f, ensure_ascii=False, indent=4)
##