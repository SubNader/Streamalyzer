from classifier.preclassifier import fetch_data
from sklearn.ensemble import RandomForestClassifier


def run(emotion):
    training_vectors, test_vectors, training_labels, test_labels = fetch_data(emotion, 'train')
    classifier = RandomForestClassifier()
    classifier.fit(training_vectors, training_labels)
    print('Random Forest Accuracy:', classifier.score(test_vectors, test_labels))
