from classifier.preclassifier import fetch_data
from sklearn.ensemble import AdaBoostClassifier


def run(emotion):
    training_vectors, test_vectors, training_labels, test_labels = fetch_data(emotion)
    learning_rates = [0.1, 0.25, 0.5, 1, 1.1, 1.25, 1.5]
    for n in learning_rates:
        classifier = AdaBoostClassifier(learning_rate=n)
        classifier.fit(training_vectors, training_labels)
        print('AdaBoost Accuracy (learning rate ', n, '):', classifier.score(test_vectors, test_labels))
