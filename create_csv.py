import pickle
import pandas as pd
import csv

with open('hate_train', 'rb') as f:
    train_data = pickle.load(f)
dict = {'text': [], 'label': []}
for data in train_data:
    dict['text'].append(data[0])
    dict['label'].append(data[1])
df = pd.DataFrame(data=dict)

df.to_csv('hate_train.csv', index=False)

with open('hate_test', 'rb') as f:
    train_data = pickle.load(f)
dict = {'text': [], 'label': []}
for data in train_data:
    dict['text'].append(data[0])
    dict['label'].append(data[1])
df = pd.DataFrame(data=dict)

df.to_csv('hate_test.csv', index=False)
