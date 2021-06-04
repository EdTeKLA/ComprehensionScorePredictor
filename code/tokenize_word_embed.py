import pandas as pd
import os
from tqdm import tqdm
import re
import numpy as np
import json

tqdm.pandas()

from gensim.models import KeyedVectors as kv
from gensim.scripts.glove2word2vec import glove2word2vec


# Some code are taken from:
# https://www.kaggle.com/alhalimi/tokenization-and-word-embedding-compatibility/data

# Make sure the current working directory is correct
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

#Build the vocabulary given a list of sentence words
def get_vocab(sentences, verbose= True):
    """
    :param sentences: a list of list of words
    :return: a dictionary of words and their frequency 
    """
    vocab={}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] +=1
            except KeyError:
                vocab[word] = 1
    return vocab

def repl(m):
    
    return '#' * len(m.group())

#Convert numerals to a # sign
def convert_num_to_pound(sentences):
    return sentences.progress_apply(lambda x: re.sub("[1-9][\d]+", repl, x)).values

# Get word embeddings
def get_embeddings(embedding_path_dict,emb_name):
    """
    :params embedding_path_dict: a dictionary containing the path, binary flag, and format of the desired embedding,
            emb_name: the name of the embedding to retrieve
    :return embedding index: a dictionary containing the embeddings"""
    
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    
    embeddings_index = {}
    
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path_dict[emb_name]['path']) if len(o)>100)    
    
    return embeddings_index

def get_emb_stats(embeddings_index):

    # Put all embeddings in a numpy matrix
    all_embs= np.stack(embeddings_index.values())

    # Get embedding stats
    emb_mean = all_embs.mean()
    emb_std = all_embs.std()
    
    num_embs = all_embs.shape[0]
    
    emb_size = all_embs.shape[1]
    
    return emb_mean,emb_std, num_embs, emb_size 

#Convert GLoVe format into word2vec format
def glove_to_word2vec(embedding_path_dict, emb_name='glove', output_emb='glove_word2vec'):
    """
    Convert the GLOVE embedding format to a word2vec format
    :params embedding_path_dict: a dictionary containing the path, binary flag, and format of the desired embedding,
            glove_path: the name of the GLOVE embedding
            output_file_path: the name of the converted embedding in embedding_path_dict. 
    :return output from the glove2word2vec script
    """
    glove_input_file = embedding_path_dict[emb_name]['path']
    word2vec_output_file = embedding_path_dict[output_emb]['path']                
    return glove2word2vec(glove_input_file, word2vec_output_file)

#find words in common between a given embedding and our vocabulary
def compare_vocab_and_embeddings(sentences, embeddings_index):
    """
    :params vocab: our corpus vocabulary (a dictionary of word frquencies)
            embeddings_index: a genim object containing loaded embeddings.
    :returns in_common: words in common,
             in_common_freq: total frequency in the corpus vocabulary of 
                             all words in common
             oov: out of vocabulary words
             oov_frequency: total frequency in vocab of oov words
    """
    in_common={}
    oov=[]
    in_common=[]
    in_common_freq = 0
    oov_freq = 0
    
    # Compose the vocabulary given the sentence tokens
    vocab = get_vocab(sentences)

    for word in tqdm(vocab):
        if word in embeddings_index:
            in_common.append(word)
            in_common_freq += vocab[word]
        else: 
            oov.append(word)
            oov_freq += vocab[word]
    
    print('Found embeddings for {:.2%} of vocab'.format(len(in_common) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(in_common_freq / (in_common_freq + oov_freq)))

    return sorted(in_common)[::-1], sorted(oov)[::-1], in_common_freq, oov_freq, vocab

# print the list of out-of-vocabulary words sorted by their frequency in teh training text
def show_oov_words(oov, vocab,  num_to_show=100):
    # Sort oov words by their frequency in the text
    sorted_oov= sorted(oov, key =lambda x: vocab[x], reverse=True )

    # Show oov words and their frequencies
    if (len(sorted_oov)>0):
        print("oov words:")
        for word in sorted_oov[:num_to_show]:
            print("%s\t%s"%(word, vocab[word]))
    else:
        print("No words were out of vocabulary.")
        
    return len(sorted_oov)

def generate_tokenized_version():
    pass

if __name__ == '__main__':
    embedding_path_dict = {
        'wiki+gigaword':{
                 'path':'../glove.6B/glove.6B.300d.txt',
                 'format': 'glove',
                 'binary': ''
                },
        'wiki_word2vec':{
                 'path':'../glove.6B/glove.6B.300d.txt.word2vec',
                 'format': 'word2vec',
                 'binary': False
        },
        'common_crawl':{
                 'path':'../glove.42B.300d.txt',
                 'format': 'glove',
                 'binary': ''
                },
        'common_word2vec':{
                 'path':'../glove.42B.300d.txt.word2vec',
                 'format': 'word2vec',
                 'binary': False
        },
    }
    emb_name = 'wiki+gigaword'
    embeddings_index= get_embeddings(embedding_path_dict, emb_name)
    import gc
    gc.collect()

    # Get embedding stats
    # emb_mean,emb_std, num_embs, emb_size = get_emb_stats(embeddings_index)
    # print("mean: %5.5f\nstd: %5.5f\nnumber of embeddings: %d\nembedding vector size:%d" \
    #     %(emb_mean,emb_std, num_embs, emb_size))
    
    df = pd.read_csv('../data/gr3_test_to_score.csv')
    sentences = df['text']
    sentences = tokenize(sentences)
    
    

    # Get words in common and out of vocabulary words
    in_common, oov, in_common_freq, oov_freq, vocab = compare_vocab_and_embeddings(sentences, embeddings_index)

    # Print a sorted list of the oov words
    show_oov_words(oov, vocab)