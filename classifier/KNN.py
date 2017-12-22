from classifier.preclassifier import fetch_data
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def run(emotion):
    training_vectors, test_vectors, training_labels, test_labels = fetch_data(emotion)
    n_neighbors = [3, 5, 7, 8, 10, 12, 15]
    accuracies = []
    for n in n_neighbors:
        classifier = KNeighborsClassifier(n_neighbors=n)
        classifier.fit(training_vectors, training_labels)
        accuracies.append(classifier.score(test_vectors, test_labels))
    plt.plot(n_neighbors, accuracies, label=emotion)
