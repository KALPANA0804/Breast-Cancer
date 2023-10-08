import numpy as np
import pandas as pd

data = pd.read_csv("C:\\Users\\KALPANA K\\Downloads\\spam.csv")

data


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:\\Users\\KALPANA K\\Downloads\\spam.csv")


X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Category'], tesimport numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_scoret_size=0.2, random_state=42)


tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)


y_pred = clf.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)


print("Accuracy:", accuracy)


data.head(6)

print(data.shape)

print(data.size)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = pd.read_csv("C:\\Users\\KALPANA K\\Downloads\\spam.csv")


X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Category'], test_size=0.2, random_state=42)


tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)


y_pred = clf.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100


plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [accuracy], color='#87CEEB', width=0.4, align='center', zorder=2)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Spam Classification Accuracy', fontsize=14)
plt.ylim(0, 100)


plt.text('Accuracy', accuracy + 2, f'{accuracy:.2f}%', ha='center', va='bottom', fontsize=12, zorder=3)

for index, value in enumerate([accuracy]):
    plt.text(index, value + 2, f'{value:.2f}%', ha='center', va='bottom', fontsize=12, color='black', zorder=3)


plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)  # Add grid lines
plt.show()

print("Accuracy:", accuracy)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


data = pd.read_csv("C:\\Users\\KALPANA K\\Downloads\\spam.csv")


X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Category'], test_size=0.2, random_state=42)


tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)


y_pred = clf.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred) * 100


confusion = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(8, 6))
plt.scatter(y_test.index, y_test, c='b', marker='o', label='Actual', alpha=0.5)
plt.scatter(y_test.index, y_pred, c='r', marker='x', label='Predicted', alpha=0.5)
plt.xlabel('Sample Index')
plt.ylabel('Category')
plt.legend(loc='upper right')
plt.title('Actual vs. Predicted Categories (Blue: Actual, Red: Predicted)')
plt.ylim([-0.5, 1.5])  # Set the y-axis limits for the categories (0 and 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.tight_layout()
plt.show()

print("Accuracy:", accuracy)


import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("C:\\Users\\KALPANA K\\Downloads\\spam.csv")

spam_count = len(data[data['Category'] == 'spam'])
ham_count = len(data[data['Category'] == 'ham'])


labels = ['Spam', 'Ham']
sizes = [spam_count, ham_count]
colors = ['#ff9999', '#66b3ff']  
explode = (0.1, 0)  

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Email Category Distribution')


plt.axis('equal') 
plt.show()


