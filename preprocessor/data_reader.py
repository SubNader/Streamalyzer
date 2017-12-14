from glob import glob
from pprint import pprint


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
    with open(path, 'rb') as f:
        # skip first line
        f.readline()
        while f:
            line = str(f.readline())
            splitted = line.split('\\t')
            print(splitted)

            # check for end of input
            if len(splitted) == 4:
                _, tweet, _, level = splitted
            else:
                break

            level = level[0]
            yield (tweet, level)


if __name__ == '__main__':
    path = get_path('joy')
    pprint([x for x in (read_tweets(path))])
