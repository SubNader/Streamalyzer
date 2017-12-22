from classifier.preclassifier import fetch_data
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def run(emotion):
    training_vectors, test_vectors, training_labels, test_labels = fetch_data(emotion)
    estimators = [10, 50, 100, 1000]
    accuracies = []
    for estimator in estimators:
        classifier = RandomForestClassifier(n_estimators=estimator)
        classifier.fit(training_vectors, training_labels)
        accuracies.append(classifier.score(test_vectors, test_labels))
        print(accuracies)
    plt.plot(estimators, accuracies, label=emotion)
