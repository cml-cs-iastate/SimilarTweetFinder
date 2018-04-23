from __future__ import division
import string
import math
import os
import pickle

from data_helper import *

with open('my_corpus_idf.pickle', 'rb') as handle:
    corpus_idf = pickle.load(handle)

with open('my_topic_dictionary.pickle', 'rb') as handle:
    topic_dictionary = pickle.load(handle)

with open('all_tokens_set.pickle', 'rb') as handle:
    all_tokens_set = pickle.load(handle)

with open('topic_tf.pickle', 'rb') as handle:
    topic_tfidf = pickle.load(handle)

with open('ground_truth_dictionary.pickle', 'rb') as handle:
    ground_truth_dictionary = pickle.load(handle)

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude


def tfidf(tweet):
    tokenize = lambda doc: doc.lower().split(" ")
    tokenized_tweet = tokenize(tweet)

    doc_tfidf = []
    for term in corpus_idf.keys():
        count = tokenized_tweet.count(term)
        if count == 0:
            tf = 0
        else:
            tf = 1 + math.log(count)
        doc_tfidf.append(tf*corpus_idf[term])

    return doc_tfidf

#Enter tweet for input
print("Enter a tweet")
tweet = input(">  ")

#Printing the first 10 tweets from topic1 in topic_dictionary
print("\nThe first 10 tweets in topic_dictionary['topic1.txt']: \n")
print(topic_dictionary["topic1.txt"][:10])

#printing out the first 15 values of corpus_idf
print("\n\ncorpus_idf:\n")
my_count = 0
should_print = True
for key, value in corpus_idf.items():
        if(should_print):
            print(key,":",value)
        if my_count > 15:
            should_print = False
        my_count += 1

#Info on the Tfidf rep of the entered tweet
print("\n\nNon-Zero Tfidf_tweet terms: \n")
tfidf_tweet = tfidf(tweet)
for i in tfidf_tweet:
    if i != 0:
        print(i)
print("\nLength of tfidf_tweet:", len(tfidf_tweet))
print("\nLength of all_tokens_set:", len(all_tokens_set),"\n\n")

#topic1 center array:
print("Topic_1 Center Array: ")
print(topic_tfidf["topic1.txt"][:10],"...",topic_tfidf["topic1.txt"][-10:])
print("Size: ", len(topic_tfidf["topic1.txt"]),"\n\n")

#Print distances between tweet and center of each topic
topic_distances = {}
for key, value in topic_tfidf.items():
    topic_distances[key] = cosine_similarity(value, tfidf_tweet)
keys = sorted(topic_distances.keys(), key=lambda k: topic_distances[k], reverse=True)
for key in keys:
    print(key,":",topic_distances[key])


#Now we test the ground truths
test_ground_truths = False
if(test_ground_truths):
    print("\n\nNow we test the ground truths: \n")

    for key1, value1 in ground_truth_dictionary.items():
        tfidf_tweet = tfidf(value1)
        topic_distances2 = {}
        for key2, value2 in topic_tfidf.items():
            topic_distances2[key2] = cosine_similarity(value2, tfidf_tweet)
        keys3 = sorted(topic_distances.keys(), key=lambda k: topic_distances2[k], reverse=True)
        for key3 in keys3:
            print(key3,":",topic_distances2[key3])

        print("^should be ", value1, "\n\n\n")

zero_terms = topic_tfidf['topic12.txt'].count(0.0)
print(zero_terms)
print("Done")
