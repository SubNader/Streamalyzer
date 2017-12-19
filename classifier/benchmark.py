from classifier import adaboost, KNN, MLP, random_forest, naive_bayes

if __name__ == '__main__':
    emotions = ['joy', 'fear', 'anger', 'sadness']
    for e in emotions:
        print('Printing accuracies for:', e)
        adaboost.run(e)
        KNN.run(e)
        MLP.run(e)
        random_forest.run(e)
        naive_bayes.run(e)
        print('-----------------------------\n')
