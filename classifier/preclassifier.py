from preprocessor import data_reader, data_processor


def fetch_data(emotion, vectorize = True):
    train_tweets_tuple = list(data_reader.read_tweets(data_reader.get_path(emotion, 'train')))
    test_tweets_tuple = list(data_reader.read_tweets(data_reader.get_path(emotion, 'dev')))
    train_tweets = [i[0] for i in train_tweets_tuple]
    train_labels = [i[1] for i in train_tweets_tuple]
    test_tweets = [i[0] for i in test_tweets_tuple]
    test_labels = [i[1] for i in test_tweets_tuple]
    if vectorize:
        train_tweets_length = len(train_tweets)
        all_tweets_vector = data_processor.vectorize(data_processor.clean(train_tweets + test_tweets), 'count')
        train_tweets = all_tweets_vector[:train_tweets_length]
        test_tweets = all_tweets_vector[train_tweets_length:]
    else:
        train_tweets = data_processor.clean(train_tweets)
        test_tweets = data_processor.clean(test_tweets)
    return train_tweets, test_tweets, train_labels, test_labels
