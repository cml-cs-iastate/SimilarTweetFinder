from __future__ import division
import string
import math
import os
import pickle



def inverse_document_frequencies(tokenized_documents):
    counter = 1
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])

    with open('all_tokens_set.pickle', 'wb') as handle:
        pickle.dump(all_tokens_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
        if(counter%100==0):
            print(counter,"/ 16886")
        counter += 1
    return idf_values

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def tf(documents):
    combined_documents = " ".join(documents)
    combined_document_tokens = combined_documents.lower().split(" ")

    doc_tfidf = []
    count = 0
    for term in corpus_idf.keys():
        tf = sublinear_term_frequency(term, combined_document_tokens)
        doc_tfidf.append((tf*corpus_idf[term])/len(documents))
        # if(count%100==0):
        #     print(count, "/", len(corpus_idf.keys()))
        # count += 1


    # temp_tf_topic = []
    # for i in range(len(all_tokens_set)):
    #     temp_tf_topic.append(0)
    #
    # for document in tokenized_documents:
    #     doc_tf = []
    #     for term in corpus_idf.keys():
    #         tf = sublinear_term_frequency(term, document)
    #         doc_tf.append(tf*corpus_idf[term])
    #     for index, value in enumerate(doc_tf):
    #         temp_tf_topic[index] += value
    #
    # for i in temp_tf_topic:
    #     i /= len(doc_tf)


    return doc_tfidf

def getFileNames(dir_path):
    fileNames = []
    for parent, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            fileNames.append(filename)

    return fileNames

def clean_str(string, Lower=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    import re
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower() if Lower else string.strip()

def getContent(filename):
    with open(filename, "r", encoding="ISO-8859-1") as f:
        lines = f.readlines()
    return lines


with open('all_tokens_set.pickle', 'rb') as handle:
    all_tokens_set = pickle.load(handle)

with open('my_corpus_idf.pickle', 'rb') as handle:
    corpus_idf = pickle.load(handle)

with open('my_topic_dictionary.pickle', 'rb') as handle:
    topic_dictionary = pickle.load(handle)

data_path_origin = './tweets/tweets/Iowa/train_data'
filenames = sorted(getFileNames(data_path_origin))
filenames = [filename for filename in filenames if filename.endswith('.txt') and not filename.startswith('topic0')]

#Build a dictionary of the tweets with each filename as key for a list of tweets in that filename
topic_dictionary = {}
#
all_tweets = []
for filename in filenames:
    lines = getContent(os.path.join(data_path_origin, filename))
    lines = [clean_str(s) for s in lines]
    topic_dictionary[filename] = lines
    for tweet in lines:
        all_tweets.append(tweet)



tokenize = lambda doc: doc.lower().split(" ")
tokenized_documents = [tokenize(d) for d in all_tweets]

corpus_idf = inverse_document_frequencies(tokenized_documents)



topic_tf = {}

for key, value in topic_dictionary.items():
    print(key)
    topic_tf[key] = tf(value)


with open('topic_tf.pickle', 'wb') as handle:
    pickle.dump(topic_tf, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('my_topic_dictionary.pickle', 'wb') as handle:
    pickle.dump(topic_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('my_corpus_idf.pickle', 'wb') as handle:
    pickle.dump(corpus_idf, handle, protocol=pickle.HIGHEST_PROTOCOL)
