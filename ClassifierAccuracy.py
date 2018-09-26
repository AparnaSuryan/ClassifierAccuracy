
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("spam.csv", encoding = 'latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], 
        axis = 1, inplace = True)
df.rename(columns = {'v1': 'labels', 'v2': 'email'}, inplace = True)
mappings = {'ham': 0, 'spam': 1}
df['labels'] = df['labels'].map(mappings)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['email'])
X = tfidf_matrix.toarray()
X_train, X_test, y_train, y_test = train_test_split(X, df['labels'], test_size=0.2)

classifiers = [KNeighborsClassifier(3), 
               DecisionTreeClassifier(), 
               SGDClassifier(), 
               LogisticRegression(),
               RandomForestClassifier(),
               SVC(),
               GaussianNB()]

accuracy_list = []
for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    accuracy_dict = {}
    accuracy_dict['Algorithm'] = name
    accuracy_dict['Accuracy'] = accuracy
    accuracy_list.append(accuracy_dict)