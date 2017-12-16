"""
Module responsible for reading tweets from data and turning them to vectors
"""


def get_path(emotion, data_pool='train'):
    """
    :param data_pool: 'train' or 'dev' 
    :param emotion: 'anger' or 'fear' or 'joy' or 'sadness'
    :return: file path for data
    """
    from glob import glob

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


def tokenize(tweet, method):
    """
    tokenizes a tweet based on certain rules
    :param tweet: a string representing the tweet
    :param method: type of tokenization
    :return: 
    """
    if method == 'normal':
        return tweet.split(' ')
    if method == 'twitter':
        from nltk.tokenize import TweetTokenizer as tokenizer
        return tokenizer().tokenize(tweet)
    else:
        raise ValueError(method + ' not available for tokenization.')


def get_stemmer(stemmer, language):
    """
        returns a stemmer
        :param method: a string representing the type of stemmer
        :return: return the stemmer object
    """
    if stemmer == 'snowball':
        from nltk.stem import SnowballStemmer
        return SnowballStemmer(language)
    else:
        raise ValueError(stemmer + ' stemmer not available.')


def clean(tweets, tokenizer='normal', stemmer='snowball', no_stopwords=True, no_mentions=True,
          remove_hashtags=True, unescape=True, remove_symbols=True, verbose=False):
    """
    cleans a list tweets 
    :param tweets: a list of tweets
    :param tokenizer: string representing the tokenizer type
    :param stemmer: type of stemming
    :param method: vectorizing type
    :param stemming: use stem of the words in a tweet
    :param no_stopwords: removes low information words if True
    :param no_mentions: removes mentions if true
    :param remove_hashtags: replaces hashtags in the form of '#a_hashtag' to 'a hashtag' 
    :param unescape: unescapes html like characters 
    :param verbose: prints the tweet after and before changes 
    :param remove_symbols: removes symbols
    :return: returns cleaned tweets
    """

    if stemmer or no_mentions or no_mentions or remove_hashtags:
        from nltk.corpus import stopwords

        stemmer = get_stemmer(stemmer, 'english')
        sw = set(stopwords.words('english')) - {'no', 'not'}

        for i, tweet in enumerate(tweets):

            tweet = tweet.lower()

            if unescape:
                from xml.sax.saxutils import unescape
                tweet = unescape(tweet)

            if remove_symbols:
                # TODO(2): change to regex to remove multiple symbols in a cleaner way
                tweet = tweet.replace(';', ' ').replace(',', ' ').replace('.', '')

            # TODO(3): remove urls

            if remove_hashtags:
                tweet = tweet.replace(' #', ' ').replace('_', ' ')

            tweet_words = tokenize(tweet, tokenizer)

            if no_mentions:
                tweet_words = [word for word in tweet_words if word and word[0] != '@']
            if no_stopwords:
                tweet_words = [word for word in tweet_words if word not in sw]
            if stemmer:
                tweet_words = [stemmer.stem(word) for word in tweet_words]

            new_tweet = ' '.join(tweet_words)

            if verbose:
                print('######', tweet)
                print("******", new_tweet)

            tweets[i] = new_tweet

    return tweets


def vectorize(tweets, method):
    """
    vectorizes tweets using a certain method
    :param tweets: a list of tweets 
    :param method: a string representing the type of vecotrizer
    :return: returns sparse matrix of vecotrized tweets
    """

    if method == 'count':
        from sklearn.feature_extraction.text import CountVectorizer
        return CountVectorizer().fit_transform(tweets)
    else:
        raise ValueError(method + ' is not available')
        ## TODO(1): add other vectorizers


if __name__ == '__main__':
    path = get_path('joy')
    twts = [x[0] for x in (read_tweets(path))]
    cleaned_twts = clean(twts, tokenizer='twitter', stemmer='snowball', verbose=True)
