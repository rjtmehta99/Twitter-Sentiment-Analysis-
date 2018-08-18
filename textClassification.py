import re
import pickle
from nltk.corpus import stopwords
from sklearn.datasets import load_files
#nltk.download('stopwords')

#Import Cornell dataset
reviews = load_files('train/')
X, y = reviews.data, reviews.target


#Storing as pickle file for faster loading of data than load_files
with open('X.pickle','wb') as f:
    pickle.dump(X,f)

with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    

#Unpickling dataset
with open('X.pickle','rb') as f:
    X = pickle.load(f)
    
with open('y.pickle','rb') as f:
    y = pickle.load(f)
    

#Preprocessing the dataset and creating a well defined corpus
corpus = []
for i in range(len(X)):
    review = re.sub(r'\W',' ',str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review)
    review = re.sub(r'\s+^[a-z]',' ',review)
    review = re.sub(r'\s+',' ',review)
    corpus.append(review)
    

#BOW Model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000, min_df=1, max_df=.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000, min_df=1, max_df=.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()



#Train test split
from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X,y,test_size = .2,random_state = 0)


#Training the dataset using logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train, sent_train)

#Prediction
sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test, sent_pred)
r2Score = classifier.score(text_test, sent_test)

#accuracy = ((cm[0][0] + cm[1][1])/400)*100
print("\n[*]THE ACCURACY IS: ",r2Score)

#Pickling the weights for twitter sentiment analysis
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
#Pickling the vectorizer
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
#Unpickling the classifier 
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)

with open('tfidfmodel.pickle','rb') as f:
     tfidf = pickle.load(f)

#sampleText = ["I suck your product"]
#sampleText = tfidf.transform(sampleText).toarray()
#print("\n[*] THE POLARITY OF YOUR STATEMENT IS: ",clf.predict(sampleText))
