import pickle
from textblob import TextBlob as tb
from utilities import contractions
from utilities import range
from utilities import PolarityBasedreducer
from utilities import SubjectivityBasedreducer
from utilities import LengthBasedadder
import numpy as np
import scipy.special

cl = pickle.load(open("trained_ML_model.sav", "rb"))

#statement1 = tb("Worst movie ever made in the history of cinemas I won't be surprised if the doesn't even cross a million maker!")
#statement2 = tb("Amazing movie but I am not happy with the plot.")
review = tb(str(input("Enter your comments about the movie :")))
for word in review.split() :
    if word.lower() in contractions :
        review = review.replace(word, contractions[word.lower()])
#print(review.sentiment)
prob = cl.predict_proba(np.array([(list(review.sentiment))]))
#print(prob[0][1])
sentiment = review.sentiment
pol = sentiment.polarity
sub = sentiment.subjectivity
#print(prob[0][1])
pf = PolarityBasedreducer(pol)
sf = SubjectivityBasedreducer(pol, sub)
la = LengthBasedadder(review, pol)
#print(la)
#print(pf)
#print(sf)
new_prob = prob[0][1]*pf*sf + la
#print(new_prob)
i = 0
while True :
    if new_prob <= range[i] :
        stars = i
        break
    i += 1
print("The stars that we gave to movie in accordance to your comments are", stars)