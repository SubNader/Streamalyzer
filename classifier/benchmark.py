from classifier import adaboost, KNN, MLP, random_forest, naive_bayes
import matplotlib.pyplot as plt

if __name__ == '__main__':
    emotions = ['joy', 'fear', 'anger', 'sadness']
    plt.title('AdaBoost learning rates vs Accuracies')
    for e in emotions:
        adaboost.run(e)
    plt.legend()
    plt.show()
    plt.title('KNN n_neighbors vs Accuracies')
    for e in emotions:
        KNN.run(e)
    plt.legend()
    plt.show()
    plt.title('MLP alphas vs Accuracies')
    for e in emotions:
        MLP.run(e)
    plt.legend()
    plt.show()
    plt.title('Random Forest estimators vs Accuracies')
    for e in emotions:
        random_forest.run(e)
    plt.legend()
    plt.show()
    for e in emotions:
        naive_bayes.run(e)
