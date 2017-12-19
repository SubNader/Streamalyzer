from classifier.preclassifier import fetch_data
from sklearn.neural_network import MLPClassifier


def run(emotion):
    training_vectors, test_vectors, training_labels, test_labels = fetch_data(emotion)
    alphas = [0.0001, 0.001, 0.01, 0.1]
    for a in alphas:
        classifier = MLPClassifier(alpha=a)
        classifier.fit(training_vectors, training_labels)
        print('MLP Accuracy (alpha ', a, '):', classifier.score(test_vectors, test_labels))
