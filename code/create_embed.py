import bcolz
import pickle
import numpy as np
import os
import json
import re

# Make sure the current working directory is correct
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def create_vocab(text_path):
    vocab = []
    sub_tests = []
    with open(text_path,'r') as fp:
        sub_test = fp.readline()
        while sub_test:
            sub_tests.append(sub_test)
            sub_test = fp.readline()

    tok_sentences = tokenize(sub_tests)
    for sentence in tok_sentences:
        for tok in sentence:
            if tok not in vocab:
                vocab.append(tok)
    
    return vocab

def tokenize(sentences):
    """tokenize sentences

    Args:
        sentences (list): list of strings
        restrict_to_len (int, optional): [description]. Defaults to -1.

    Returns:
        list: list of list of tokens
    """
    
    tok_sentences = [re.findall(r"[\w]+[']*[\w]+|[\w]+|[.,!?;]", x ) \
                        for x in sentences]
                        
    with open('contractions.json', 'r') as fp:
        contractions = json.load(fp)

    for i, sentence in enumerate(tok_sentences):
        new_sentence = []
        for tok in sentence:
            new_tok = tok.lower()
            if new_tok in contractions:
                for word in contractions[new_tok].split():
                    new_sentence.append(word)
            else:
                if new_tok.endswith("'s"):
                    new_tok = new_tok[:-2]
                new_sentence.append(new_tok)
        tok_sentences[i] = new_sentence

    return tok_sentences

glove_path = '../glove.6B'

######################
# Generate glove vectors pickle file
######################
# words = []
# idx = 0
# word2idx = {}
# vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.300d.dat', mode='w')

# with open(f'{glove_path}/glove.6B.300d.txt', 'rb') as f:
#     for l in f:
#         line = l.decode().split()
#         word = line[0]
#         words.append(word)
#         word2idx[word] = idx
#         idx += 1
#         vect = np.array(line[1:]).astype(np.float)
#         vectors.append(vect)
    
# vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir=f'{glove_path}/6B.300d.dat', mode='w')
# vectors.flush()
# pickle.dump(words, open(f'{glove_path}/6B.300d.pkl', 'wb'))
# pickle.dump(word2idx, open(f'{glove_path}/6B.300d_idx.pkl', 'wb'))

######################
# Load glove embedding
######################
vectors = bcolz.open(f'{glove_path}/6B.300d.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.300d.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.300d_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

target_vocab = create_vocab("../subtest_txt/gr3_paragraphs.txt")

matrix_len = len(target_vocab)
emb_dim = 300
weights_matrix = np.zeros((matrix_len, emb_dim))
words_found = 0

for i, word in enumerate(target_vocab):
    try: 
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))


pickle.dump(weights_matrix, open('vocab_embedding.pkl', 'wb'))
print(weights_matrix)
print(weights_matrix.shape)

