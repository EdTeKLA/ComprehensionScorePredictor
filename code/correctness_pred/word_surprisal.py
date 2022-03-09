import string
import pickle
from statistics import mean, stdev
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.lm.preprocessing import padded_everygram_pipeline,pad_both_ends
from nltk.lm import MLE

from split_into_sentences import split_into_sentences



def preprocess(corpus):
    corpus = corpus.lower()

    corpus = corpus.replace('_',' ')
    corpus = "".join([char for char in corpus if char not in string.punctuation])

    words = word_tokenize(corpus)

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]


    return lemmatized

def get_surprisal_metric(sub_test,lm):
    sentences = split_into_sentences(sub_test)
    text = []
    mean_surprisal = []
    for s in sentences:
        new = preprocess(s)
        new = pad_both_ends(new, n=2)
        text.append(list(new))
    for t in text:
        total = 0
        count = 0
        for i in range(1,len(t)-1): # skip iterating the padding
            total += lm.score(t[i],[t[i-1]])
            count += 1
            print(lm.score(t[i],[t[i-1]]))
        mean_surprisal.append(total/count)

    return [mean(mean_surprisal), stdev(mean_surprisal), max(mean_surprisal)]
    
    

if __name__ == '__main__':
    '''
    Below is script that combines all three training data file into one

    filenames = ['../../data/cbt_test.txt', '../../data/cbt_train.txt', '../../data/cbt_valid.txt']
    with open('../../data/cbt_whole.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
            
    '''
    
    sentences = pickle.load(open('sentences.pkl','rb'))

    text = []
    for s in sentences:
        text.append(preprocess(s))

    train, vocab = padded_everygram_pipeline(2, text)
    lm = MLE(2)
    lm.fit(train, vocab)
    print(lm.vocab)
    print(lm.score('test'))

    sub_tests = []

    corpus_path = f"../../subtest_txt/gr3_paragraphs.txt"
    with open(corpus_path,'r') as fp:
        sub_test = fp.readline()
        while sub_test:
            sub_tests.append(sub_test)
            sub_test = fp.readline()

    subtext_surp = []
    for sub in sub_tests:
        subtext_surp.append(get_surprisal_metric(sub,lm))
    
    print(subtext_surp)
    pickle.dump(subtext_surp,open('surprisal.pkl','wb'))

