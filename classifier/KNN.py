from classifier.preclassifier import fetch_data
from sklearn.neighbors import KNeighborsClassifier


def run(emotion):
    training_vectors, test_vectors, training_labels, test_labels = fetch_data(emotion, 'train')
    classifier = KNeighborsClassifier()
    classifier.fit(training_vectors, training_labels)
    print('KNN Accuracy:', classifier.score(test_vectors, test_labels))
