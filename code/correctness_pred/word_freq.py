import string
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from collections import Counter

from matplotlib import pyplot as plt

def preprocess(corpus):
    corpus = corpus.lower()

    corpus = corpus.replace('_',' ')
    corpus = "".join([char for char in corpus if char not in string.punctuation])

    words = word_tokenize(corpus)

    # print(words[:20])

    stop_words = stopwords.words('english')
    # print(stop_words)
    filtered_words = [word for word in words if word not in stop_words]

    # print(filtered_words[:20])

    
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]

    # porter = PorterStemmer()
    # stemmed = [porter.stem(word) for word in lemmatized]

    return lemmatized

if __name__ == '__main__':
    with open("../../data/cbt_whole.txt",'r') as file:
        corpus = file.read().replace('\n', ' ')
    words = preprocess(corpus)
    freq = Counter(words)

    print('vocab size:', len(freq))
    # for key in list(freq.keys())[0:10]:
    #     print(key, freq[key])

    ranking = sorted(freq.items(),key=lambda item:item[1],reverse=True)

    threshold = ranking[700][1]
    rare_words = []
    wordlist = []
    data_pts = []

    for i in range(0,len(ranking)):
        value = ranking[i][1]
        data_pts.append(value)

    for i in range(len(ranking)):
        if ranking[i][1] <= threshold:
            rare_words.append(ranking[i][0])
        else:
            wordlist.append(ranking[i][0])

    print("First 50 rare words:\n", rare_words[:50])
    print("Last 50 common words\n", wordlist[-50:])
    print("Number of words in common words list:\n", len(wordlist))
    print("Number of words in rare words list:\n", len(rare_words))

    with open("wordlist.pkl",'wb') as fp:
        pickle.dump(wordlist, fp)

    plt.plot(data_pts)
    plt.axvline(x=700, label='Threshold at x = {}'.format(700), c='r', linestyle = '--')
    plt.title("Children's Book Test Word Frequency")
    plt.xlabel('Word Index')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # lemmatizer = WordNetLemmatizer()
    # print(lemmatizer.lemmatize('corpora'))
    # print(lemmatizer.lemmatize('better'))