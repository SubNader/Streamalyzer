"""
Module responsible for reading tweets from dataset
"""


def get_path(emotion, data_pool='train'):
    """
    :param data_pool: 'train' or 'dev'
    :param emotion: 'anger' or 'fear' or 'joy' or 'sadness'
    :return: file path for data
    """
    from glob import glob
    result = glob('../{}_dataset/*-{}-{}.txt'.format(data_pool, emotion, data_pool))

    return result[0]


def read_tweets(path):
    """
    generates tweets from path 
    :param path: string 
    :return: generator object to be used to iterate over data=[(tweet,level)]
    """
    with open(path, 'rb') as f:
        # skip first line
        f.readline()
        while f:
            line = str(f.readline())
            splitted = line.split('\\t')

            # check for end of input
            if len(splitted) == 4:
                _, t, _, level = splitted
            else:
                break

            level = level[0]
            yield (t, level)


# TODO(4): Remove this function
def read_all_tweets(data_pool='train'):
    """
    reads all tweets from a dataset
    :param data_pool: a string representing the data pool  
    :return: returns a tuple of tweets and corresponding labels
    """
    emotions = ['anger', 'fear', 'joy', 'sadness']
    tweets = []
    labels = []
    for i, emotion in enumerate(emotions):
        path = get_path(emotion, data_pool)
        # noinspection PyTypeChecker
        emotion_tweets = [t[0] for t in read_tweets(path)]
        tweets += emotion_tweets
        labels += [i] * len(emotion_tweets)

    return tweets, labels
