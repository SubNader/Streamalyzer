from classifier.preclassifier import fetch_data
from sklearn.naive_bayes import GaussianNB


def run(emotion):
    training_vectors, test_vectors, training_labels, test_labels = fetch_data(emotion)
    classifier = GaussianNB()
    classifier.fit(training_vectors.toarray(), training_labels)
    print('Naive Bayes Accuracy:', classifier.score(test_vectors.toarray(), test_labels))
