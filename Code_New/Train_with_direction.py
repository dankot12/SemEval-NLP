

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# opening the raw training file provided

train = pd.read_csv("CSV Files/Train_data_direction.csv", header=0, delimiter=",")


def sentence_to_words( raw_review ):
    # Function to convert a raw sentence to a string of words
    # The input is a single string (a raw sentences), and
    # the output is a single string (a preprocessed sentence)
    review_text = raw_review
    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # searching a set is much faster than searching
    # convert the stop words to a set
    # stops = set(stopwords.words("english"))
    # Here,even stop words matter, so  wont remove them.
    stops = []
    #
    # creating a list of words
    meaningful_words = [w for w in words if not w in stops]
    #
    # Join the words back into one string separated by space,
    # and return the result.
    return " ".join( meaningful_words )


# Initialize an empty list to hold the clean reviews
clean_train = []

# Loop over each sentence; create an index i that goes from 0 to the length
# of the sentence list
for i in range(8000):
    # Call our function for each one, and add the result to the list of
    # clean sentences
    clean_train.append(sentence_to_words( train.iloc[i][0] ) )


print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features =40000,ngram_range=(1,4))

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(clean_train)

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()
#train_data_features = sklearn_representation.toarray()

#To see the bag of words:
#vocab = vectorizer.get_feature_names()
#print vocab

test = pd.read_csv("CSV Files/Test_data.csv", header=0, delimiter=",")

# Verify that there are 2717 rows and 1 column
print "The shape of test data is",test.shape
print

clean_test = []

print "Cleaning and parsing the test set sentences...\n"
for i in range(len(test)):
    if( (i+1) % 1000 == 0 ):
        print "Cleaning %d of %d\n" % (i+1, len(test))
    clean_review = sentence_to_words( test.iloc[i][0] )
    clean_test.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test)
test_data_features = test_data_features.toarray()

keys = pd.read_csv("CSV Files/Test_Keys_direction.csv", header=0, delimiter="\t")

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model


ML_ALGO = {
#"RANDOM FOREST" : RandomForestClassifier(n_estimators = 500),
#"DECISION TREE" : tree.DecisionTreeClassifier(),
#"SVM" : svm.SVC(kernel='poly', C=1, gamma=1),
#"K NEAREST NEIGHBOR" : KNeighborsClassifier(n_neighbors=20),
"LOGISTIC REGRESSION" : LogisticRegression(),
#"NAIVE BAYES" : GaussianNB()
}

#test the various algorithms
result = {}
print("Testing ML ALgorithm's...")
for key,alg in ML_ALGO.items():
    print key,alg
    clf = alg
    clf.fit(train_data_features, train[['Category']].values.ravel())
    prediction = clf.predict(test_data_features)
    count = 0
    for i in range(len(test)):
        if prediction[i] == keys['Category'][i]:
            count += 1
    accuracy = 100.00*count/len(test)
    print("\n" + str(key) + "'s Accuracy:    %.2f%%" % (accuracy * 1.00))
    result[key] = accuracy

#find the best algorithm in terms of accuracy
Best = max(result, key = result.get)
print("\nThe best algorithm for this problem is %s with an accuracy of %f%%." % (Best,result[Best]))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# genrate the confusion matrix
print "The confusion matrix is"
print confusion_matrix(keys['Category'], prediction)
print "The f1 score is ",f1_score(keys['Category'], prediction,average='macro')
