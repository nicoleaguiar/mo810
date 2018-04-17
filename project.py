import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer


def stem_tokens(tokens, stemmer):
    st = [ ]
    for token in tokens:
        st.append(stemmer.stem(token))
    return st


def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, PorterStemmer())
    return stems

def pre_processing(data):
    count_vec = CountVectorizer(tokenizer = tokenize, analyzer = 'word', stop_words = 'english', binary = True)
    print(count_vec)
    x = count_vec.fit_transform(data)
    return x
    
def bag_of_words(data):
    tf_transformer = TfidfTransformer().fit(data)
    x_tf = tf_transformer.transform(data)


name_file = '/home/nicole/Documentos/mo810/Musical_Instruments_5.json'
data = pd.read_json(name_file, lines = True, orient = 'records')
data = data.drop(data.columns[[0,1,4,6,7,8]], axis = 1)
print(data)
processed_data = pre_processing(data['reviewText'])
bag_of_words(processed_data)
