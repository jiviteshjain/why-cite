import re
import numpy as np
import jsonlines

file_names = [
    "dev.jsonl",
    "train.jsonl",
    "test.jsonl"
]

def process_json(data):

    extended = data['sectionName']
    print(extended)

    try:

        if("discussion" in extended.lower()): 
            print("yes")
            data['sectionName'] = "discussion"

        elif("introduction" in extended.lower()): 
            print("yes")
            
            data['sectionName'] = "introduction"
        
        elif("related work" in extended.lower()): 
            print("yes")

            data['sectionName'] = "related work"

        elif("method" in extended.lower()): 
            print("yes")

            data['sectionName'] = "methods"

        elif("experiments" in extended.lower()): 
            data['sectionName'] = "experiments"
        
        elif("results" in extended.lower()): 
            data['sectionName'] = "results"
        
        elif("conclusion" in extended.lower()): 
            data['sectionName'] = "conclusion"
        
        else: data['sectionName'] = None
    
    except:
        data['sectionName'] = None

    # extended = data['sectionName']
    # print(extended.lower())

    return data

if __name__ == '__main__':

    for file_name in file_names:
        instances = list(jsonlines.open(f"../../data/scicite/{file_name}"))
        new_instances = list(map(process_json, instances))

        f = open(f"../../data/scicite/processed_{file_name}", "w")
        writer = jsonlines.Writer(f)
        writer.write_all(new_instances)
        f.close()
