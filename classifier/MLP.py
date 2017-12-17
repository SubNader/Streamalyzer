from classifier.preclassifier import fetch_data
from sklearn.neural_network import MLPClassifier


def run(emotion):
    training_vectors, test_vectors, training_labels, test_labels = fetch_data(emotion)
    classifier = MLPClassifier()
    classifier.fit(training_vectors, training_labels)
    print('MLP Accuracy:', classifier.score(test_vectors, test_labels))
