from glob import glob


def get_path(emotion, data_pool='train'):
    """
    :param data_pool: 'train' or 'dev' 
    :param emotion: 'anger' or 'fear' or 'joy' or 'sadness'
    :return: file path for data
    """

    result = glob('../{}_dataset/*-{}-{}.txt'.format(data_pool, emotion, data_pool))

    assert len(result) == 1
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
                _, tweet, _, level = splitted
            else:
                break

            level = level[0]
            yield (tweet, level)


def vectorize(tweets, method='count', stemming=True, no_stopwords=True, no_mentions=True):
    """
    vectorizes a list tweets 
    :param tweets: a list of tweets 
    :param method: vectorizing type
    :param stemming: use stem of the words in a tweet
    :param no_stopwords: removes low information words if True
    :param no_mentions: removes mentions if true
    :return: returns a sparse matrix of vectorized tweets
    """

    def __count_vectorize(tweets):
        from sklearn.feature_extraction.text import CountVectorizer
        return CountVectorizer().fit_transform(tweets)

    if stemming or no_mentions or no_mentions:
        from nltk.stem import SnowballStemmer
        from nltk.corpus import stopwords

        stemmer = SnowballStemmer('english')
        sw = stopwords.words('english')
        for i, tweet in enumerate(tweets):
            tweet_words = tweet.split(' ')
            if stemming:
                tweet_words = [stemmer.stem(word) for word in tweet_words]
            if no_mentions:
                tweet_words = [word for word in tweet_words if word and word[0] != '@']
            if no_stopwords:
                tweet_words = [word for word in tweet_words if word not in sw]
            tweets[i] = ' '.join(tweet_words)

    if method == 'count':
        return __count_vectorize(tweets)

        ## TODO(1): add other vectorizers


if __name__ == '__main__':
    path = get_path('joy')
    tweets = [x[0] for x in (read_tweets(path))]
    print(vectorize(tweets).shape)
