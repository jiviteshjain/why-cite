from sklearn.utils.class_weight import compute_class_weight

import jsonlines
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


counts = defaultdict(int)
y = []

df = pd.read_csv('./train_ns_extended.csv')

        
labels = [0, 1, 2, 3, 4, 5]

print(labels)
    
print(compute_class_weight(class_weight='balanced', classes=labels, y=df['citation_class_label']))
