from gensim.models import KeyedVectors
import re
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from encoder import QWE_ENGLISH
import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd
import os
import random


data_idx = 5

try:
    os.mkdir('result-data_idx_{}'.format(data_idx))
except:
    pass

# Load the original model
model_path = 'word_embedding_bank/data_idx_{}/model.bin'.format(data_idx)
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
# Select 1000 random words from the original model
random_words = random.sample(model.index_to_key, 3000)
# Create a new model with embeddings for these random words



word_pairs = []
we_dist = []
qe_dist = []

for word_i in tqdm.tqdm(random_words):
    df = pd.DataFrame([])
    key_i = word_i
    word_i = word_i.split('_')[0]
    if word_i.isalpha():
        for word_j in model.index_to_key:
                key_j = word_j
                word_j = word_j.split('_')[0]
                if len(word_j) == len(word_i) and word_j.isalpha():
                    try:
                        word_i = word_i.lower()
                        word_j = word_j.lower()
                        # Quantum Encoding
                        _ ,qe_i = np.abs(QWE_ENGLISH(word_i))
                        _ ,qe_j = np.abs(QWE_ENGLISH(word_j))
                        #print(qe_i)
                        #print(qe_j)
                        qe_i = qe_i.flatten().reshape(1, -1)
                        qe_j = qe_j.flatten().reshape(1, -1)
                        # Classical Encoding
                        we_i = model[key_i].reshape(1, -1)
                        we_j = model[key_j].reshape(1, -1)
                        # Result
                        qe_dist.append(euclidean(qe_i,qe_j))
                        we_dist.append(euclidean(we_i, we_j))
                        word_pairs.append('{}-{}'.format(word_i, word_j))
                    except:
                        pass

df['word_pairs'] = word_pairs
df['we_dist'] = we_dist
df['qe_dist'] = qe_dist
df.to_csv('result-data_idx_{}/dist-data_idx_{}.csv'.format(data_idx,data_idx), index = False)



