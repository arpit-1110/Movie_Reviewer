from textblob.classifiers import NaiveBayesClassifier
import os
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from textblob import TextBlob as tb
train = np.zeros((25000, 2))
directory = os.fsencode("A:/aclImdb/train/pos")
# # # # #
i = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
#     #print(filename)
    f = open("A:/aclImdb/train/pos/" + filename, "r", encoding="utf8")
    com = tb(f.read())
# #     allWords = nltk.tokenize.word_tokenize(com)
# #     allWords = [w.lower() for w in allWords]
# #     allWordsDist = nltk.FreqDist(w.lower() for w in allWords)
# # # #
# #     mostCommon = allWordsDist.most_common(5)
# #     mostCommon = dict(mostCommon)
# #     ls_keys = list(mostCommon.keys())
# #     for i in range(0, len(ls_keys)):
# #         while ls_keys[i] in allWords:
# #             allWords.remove(ls_keys[i])
# #     com = ' '.join(allWords)
# # #
    train[i] = np.array(list(com.sentiment))
    i += 1
# # # # #
directory = os.fsencode("A:/aclImdb/train/neg")
# # # # #
for file in os.listdir(directory):
    filename = os.fsdecode(file)
# # #     #print(filename)
    f = open("A:/aclImdb/train/neg/" + filename, "r", encoding="utf8")
    com = tb(f.read())
# #     allWords = nltk.tokenize.word_tokenize(com)
# #     allWords = [w.lower() for w in allWords]
# #     allWordsDist = nltk.FreqDist(w.lower() for w in allWords)
# # #
# #     mostCommon = allWordsDist.most_common(5)
# #     mostCommon = dict(mostCommon)
# #     ls_keys = list(mostCommon.keys())
# #     for i in range(0, len(ls_keys)):
# #         while ls_keys[i] in allWords:
# #             allWords.remove(ls_keys[i])
# #     com = ' '.join(allWords)
#     #print(filename)
    train[i] = np.array(list(com.sentiment))
    i += 1

cl = MLPClassifier(solver='lbfgs',activation='logistic',
                     alpha=1e-2, hidden_layer_sizes=(10,5),
                     momentum=0.25, random_state=1)
# #cl = NaiveBayesClassifier(train)
y = np.zeros(25000)
i = 0
while i != 12500 :
    y[i] = 1
    i += 1
#
#sampler = RBFSampler()
#train = sampler.fit_transform(train)
cl.fit(train, y)
filename = 'trained_ML_model_ver4.sav'

pickle.dump(cl, open(filename, 'wb'))

