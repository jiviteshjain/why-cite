import pandas
import copy
import json
import random


with open('3c_para.txt', "r") as f:
    lines = [line.strip() for line in f.readlines()]


df = pandas.read_csv('./train_newsplit.csv')

i = 0
for ind in df.index:
    if df['citation_class_label'][ind] != 0 or random.random() < 0.2:
        df.loc[len(df.index)] = [len(df.index), f'CC{len(df.index)}', df['core_id'][ind],
                                 df['citing_title'][ind], df['citing_author'][ind],
                                 df['cited_title'][ind], df['cited_author'][ind],
                                 lines[i], df['citation_class_label'][ind]]
    i += 1

df.to_csv('3c_para_train.csv')
