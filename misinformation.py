import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


dataset = pd.read_excel('newsfinal.xlsx', sheet_name=0)
dataset.head()

titles = dataset['Title'].tolist()
#articles = df['Article'].tolist()
label = dataset['Type'].tolist()

tokenizer = Tokenizer(titles)
tokenizer.encode()

for i in range(0,6):
    print(type(titles[i])," ",titles[i])
