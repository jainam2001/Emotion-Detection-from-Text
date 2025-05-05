# Import necessary libraries
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

# Function to read and parse the data from file
def read_data(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())  # extract label from line
            text = line[line.find("]") + 1:].strip()  # extract text from line
            data.append([label, text])
    return data

# Load the data
file = 'text.txt'
data = read_data(file)
print("Number of instances: {}".format(len(data)))

# Function to create n-grams from tokens
def ngram(token, n):
    output = []
    for i in range(n - 1, len(token)):
        ngram = ' '.join(token[i - n + 1:i + 1])
        output.append(ngram)
    return output

# Function to extract features from text
def create_feature(text, nrange=(1, 1)):
    text_features = []
    text = text.lower()
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)  # remove special characters
    for n in range(nrange[0], nrange[1] + 1):
        text_features += ngram(text_alphanum.split(), n)
    text_punc = re.sub('[a-z0-9]', ' ', text)  # keep punctuation only
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)  # return feature count dictionary

# Test feature extraction
print(create_feature("aly wins the gold gold"))

# Convert multi-label binary vector to emotion labels
def convert_label(item, name):
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)):
        if items[idx] == 1:
            label += name[idx] + " "
    return label.strip()

# Define emotion labels
emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

# Extract features and labels from data
X_all = []
y_all = []
for label, text in data:
    y_all.append(convert_label(label, emotions))
    X_all.append(create_feature(text, nrange=(1, 4)))

print(X_all[0])
print(y_all[0])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=123)

# Train and evaluate classifier
def train_test(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return train_acc, test_acc

# Vectorize feature dictionaries
vectorizer = DictVectorizer(sparse=True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Define classifiers
lsvc = LinearSVC(random_state=123)
clifs = [lsvc]

# Train and test classifiers
print("| {:25} | {} | {} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
print("| {} | {} | {} |".format("-" * 25, "-" * 17, "-" * 13))
for clf in clifs:
    clf_name = clf.__class__.__name__
    train_acc, test_acc = train_test(clf, X_train, X_test, y_train, y_test)
    print("| {:25} | {:17.7f} | {:13.7f} |".format(clf_name, train_acc, test_acc))

# Count frequency of each label
label_freq = {}
for label, _ in data:
    label_freq[label] = label_freq.get(label, 0) + 1

# Print label frequencies
for l in sorted(label_freq, key=label_freq.get, reverse=True):
    print("{:10}({})  {}".format(convert_label(l, emotions), l, label_freq[l]))

# Emoji dictionary for emotion prediction display
emoji_dict = {
    "joy": "😂 - Happy",
    "fear": "😱 - Fear",
    "anger": "😠 - Angry",
    "sadness": "😢 - Sad",
    "disgust": "😒 - Disgust",
    "shame": "🨭 - Shame",
    "guilt": "😳 - Guilt"
}

# User input and prediction
t1 = input("Enter the sentence : ")
texts = [t1]
for text in texts:
    features = create_feature(text, nrange=(1, 4))
    features = vectorizer.transform(features)
    prediction = clf.predict(features)[0]  # predict emotion
    print(emoji_dict[prediction])  # display emoji result

