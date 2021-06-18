import gensim
import re
import json

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

def main():
    sentences = []
    with open("../data/cbt_train.txt",'r') as fp:
        sentences = fp.readlines()
    
    sentences_token = tokenize(sentences)
    model = gensim.models.Word2Vec(sentences=sentences_token)
    print(sentences_token[:5])

if __name__ == '__main__':
    main()
