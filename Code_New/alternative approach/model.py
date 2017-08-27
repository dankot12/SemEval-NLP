
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# import training data
data = pd.read_csv('Train_data.csv')

# represent term frequency within N-gram ranging from 1 to 7
vectorizer = TfidfVectorizer(ngram_range=(1,7))
# transform the vectorizer to obtain a vector
vec = vectorizer.fit_transform(data['Sentence'])

# function to calculate the class of any given sentence
def get_class(q):
    # transform the vectorizer to obtain a vector
    my_q = vectorizer.transform([q])
    # use the cosine formula to find similarity between the two vectors
    cs = cosine_similarity(my_q, vec)
    # sort the result obtained in ascending order based on the extent to which they match
    rs = pd.Series(cs[0]).sort_values(ascending=0)
    rsi = rs.index[0]
    return data.iloc[rsi]['Category']


# import the test data and its class
test=pd.read_csv('test.csv')
key=pd.read_csv('test_class.csv')

# preprocess the test data inorder to match the format of the input data
for str in test['Sentence']:
    str=str[4:len(str)]

arr=[]
for str in key['Category']:
    arr.append(str[5:len(str)])
count=0
i=0

# compare the classes obtained and the actual classes
for str,key in zip(test['Sentence'],arr):
    if(get_class(str)==key):
        count=count+1;
print(count)

# print the acuuracy value
print(count*100/len(arr))

