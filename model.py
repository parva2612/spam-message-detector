from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import pickle
import re
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
mnb = MultinomialNB(alpha=0.8)
cv = CountVectorizer(max_features=3500)


# IMPORTING CSV
spam_df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
spam_df = spam_df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
z = spam_df["v1"]
y = spam_df["v2"]
print("IMPORTED DATASET:")
print(spam_df.head())

# function corpus
corpus = []
for i in range(0, len(spam_df)):
    review = re.sub('[^a-zA-Z]', ' ', spam_df['v2'][i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word)
              for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# vocab
X = cv.fit_transform(corpus).toarray()
pickle.dump(cv, open('cv-transform.pkl', 'wb'))

# model
y = pd.get_dummies(spam_df['v1'])
y = y.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)
mnb.fit(X_train, y_train)
pickle.dump(mnb, open("model-spam.pkl", 'wb'))
