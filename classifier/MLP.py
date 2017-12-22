from classifier.preclassifier import fetch_data
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def run(emotion):
    training_vectors, test_vectors, training_labels, test_labels = fetch_data(emotion)
    alphas = [0.0001, 0.001, 0.01, 0.1]
    accuracies = []
    for a in alphas:
        classifier = MLPClassifier(alpha=a)
        classifier.fit(training_vectors, training_labels)
        accuracies.append(classifier.score(test_vectors, test_labels))
    plt.plot(alphas, accuracies, label=emotion)
