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


if __name__ == '__main__':
    from preprocessor.data_processor import *
    path = get_path('joy')
    twts = [x[0] for x in (read_tweets(path))]
    cleaned_twts = clean(twts, tokenizer='twitter', stemmer='snowball', verbose=True)
