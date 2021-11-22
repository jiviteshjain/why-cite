import pandas as pd

df = pd.read_csv('train_ns_extended.csv')

out_file = open('3c_extended.txt', 'w')

lines = [x + '\n' for x in df['citation_context']]

for line in lines:
    out_file.write(line)
