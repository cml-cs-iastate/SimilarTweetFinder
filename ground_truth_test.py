import csv
import pickle

with open('new_ground_truth.csv', newline='') as csvfile:
    tweets = csv.reader(csvfile)
    ground_truth_dictionary = {}
    for row in tweets:
        ground_truth_dictionary[row[0]] = row[2]

    for key, value in ground_truth_dictionary.items():
        print(key,":",value)


with open('ground_truth_dictionary.pickle', 'wb') as handle:
    pickle.dump(ground_truth_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
