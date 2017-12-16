from classifier.preclassifier import fetch_data
from sklearn.ensemble import AdaBoostClassifier


def run(emotion):
    training_vectors, test_vectors, training_labels, test_labels = fetch_data(emotion, 'train')
    classifier = AdaBoostClassifier()
    classifier.fit(training_vectors, training_labels)
    print('AdaBoost Accuracy:', classifier.score(test_vectors, test_labels))
