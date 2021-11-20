import jsonlines
from collections import defaultdict
from tqdm import tqdm

counts = defaultdict(int)
with jsonlines.open('./augmented_acl_processed_train2.jsonl') as read_file:
    for obj in tqdm(read_file):
        counts[obj['intent']] += 1
    
s = sum([counts[key] for key in counts.keys()])
for key in counts.keys():
    print(key, counts[key]/s)
