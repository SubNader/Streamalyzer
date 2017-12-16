from classifier.preclassifier import fetch_data
from sklearn.ensemble import RandomForestClassifier


def run():
    tweet_vectors, labels = fetch_data('joy', 'train')
    training_vectors = tweet_vectors[0:tweet_vectors.shape[0] // 2]
    training_labels = labels[0:len(labels) // 2]
    test_vectors = tweet_vectors[tweet_vectors.shape[0] // 2:]
    test_labels = labels[len(labels) // 2:]
    classifier = RandomForestClassifier()
    classifier.fit(training_vectors, training_labels)
    print('Random Forest Accuracy:', classifier.score(test_vectors, test_labels))
