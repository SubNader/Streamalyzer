from classifier.preclassifier import fetch_data
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt


def run(emotion):
    training_vectors, test_vectors, training_labels, test_labels = fetch_data(emotion)
    learning_rates = [0.1, 0.25, 0.5, 0.7, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.1, 2.25, 2.5, 3, 3.1, 3.5]
    accuracies = []
    for n in learning_rates:
        classifier = AdaBoostClassifier(learning_rate=n)
        classifier.fit(training_vectors, training_labels)
        accuracies.append(classifier.score(test_vectors, test_labels))
    plt.plot(learning_rates, accuracies, label=emotion)
