from preprocessor import data_reader, data_processor


def fetch_data(emotion, data_pool='train'):
    tweets_labels_tuple = list(data_reader.read_tweets(data_reader.get_path(emotion, data_pool)))
    tweets = [i[0] for i in tweets_labels_tuple]
    labels = [i[1] for i in tweets_labels_tuple]
    return data_processor.vectorize(data_processor.clean(tweets), 'count'), labels
