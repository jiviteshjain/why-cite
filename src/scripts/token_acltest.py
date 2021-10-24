import re
import numpy as np
import jsonlines

file_names = [
    "dev.jsonl",
    "train.jsonl",
    "test.jsonl"
]

def process_json(data):
    extended = data['extended_context']

    new_text = extended.split(' ')
    
    return len(new_text)

if __name__ == '__main__':

    for file_name in file_names:
        instances = list(jsonlines.open(f"../../data/acl-arc/processed_{file_name}"))
        new_instances = list(map(process_json, instances))

        print(np.min(new_instances))
        # print(new_instances[0]['extended_context'])