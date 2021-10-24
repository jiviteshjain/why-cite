import re
import jsonlines

file_names = [
    "dev.jsonl",
    "train.jsonl",
    "test.jsonl"
]

def process_json(data):
    text = data['text']
    extended = data['extended_context']
    cleaned = data['cleaned_cite_text']

    new_text = extended.replace(text, cleaned)

    data['extended_context'] = new_text
    return data

if __name__ == '__main__':

    for file_name in file_names:
        instances = list(jsonlines.open(f"../../data/acl-arc/{file_name}"))
        new_instances = list(map(process_json, instances))

        f = open(f"../../data/acl-arc/processed_{file_name}", "w")
        writer = jsonlines.Writer(f)
        writer.write_all(new_instances)
        f.close()

        # print(new_instances[0]['extended_context'])