def split(data, labels, test_size=0.5, random_state=95, shuffle=False):
    """
    splits the data using a certain ratio
    :param data: a list representing the data set
    :param test_size: the test set size (range 0.0:1.0)
    :param random_state: RandomState instance
    :param shuffle: whether the data should be shuffled or not
    :return: training set, training labels, test set and test labels.
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(data, labels, test_size=test_size, random_state=random_state, shuffle=shuffle)


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
        :param stemmer: a string representing the type of stemmer
        :return: return the stemmer object
    """
    if stemmer == 'snowball':
        from nltk.stem import SnowballStemmer
        return SnowballStemmer(language)
    else:
        raise ValueError(stemmer + ' stemmer not available.')


def clean(tweets, tokenizer='normal', stemmer='snowball', no_stopwords=True, no_mentions=True,
          remove_hashtags=True, unescape=True, remove_symbols=True, remove_urls=True, verbose=False):
    """
    cleans a list tweets 
    :param tweets: a list of tweets
    :param tokenizer: string representing the tokenizer type
    :param stemmer: type of stemming
    :param no_stopwords: removes low information words if True
    :param no_mentions: removes mentions if true
    :param remove_hashtags: replaces hashtags in the form of '#a_hashtag' to 'a hashtag' 
    :param unescape: unescapes html like characters 
    :param remove_symbols: removes symbols
    :param verbose: prints the tweet after and before changes 
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

            if remove_urls:
                tweet = remove_url(tweet)

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

def remove_url(tweet):
    """
    remove urls from a tweet
    :param tweet: an instance of a tweet
    :return: returns the instance after replacing the url(s) with space character(s)
    """

    import re
    url_regex = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.]\(?:com|net|org|edu|gov|mil|aero|asia|biz|cat\|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
    return re.sub(url_regex, ' ', tweet)