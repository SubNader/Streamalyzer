class Classifier:

    def __init__(self):
        self.model_name = None
    
    def train(self, training_set, training_labels, model_name='ft_model'):
        """
        trains a supervised model on the specified training set
        :param training_set: the training set
        :param training_labels: the labels vector of the training set
        :param model_name: custom model output filename
        """

        training_file_path = 'ft_extras/ft_training_data.txt'
        training_file = open(training_file_path, 'w')
        for index, tweet in enumerate(training_set):
            ft_tweet = '__label__' + training_labels[index] + ' ' + tweet
            training_file.write("%s\n" % ft_tweet)

        from pyfasttext import FastText
        ft_object = FastText()
        # TODO: handle parametrized epoch and learning rate values
        ft_model = ft_object.supervised(input=training_file_path, output=('ft_extras/'+model_name))
        self.model_name = model_name

    def predict(self, test_set, test_labels_vector=None, report_accuracy=True):
        """
        uses the trained model to predict the test set
        :param test_set: the test set
        :param test_labels_vector: the labels vector of the test set for accuracy computation
        :param report_accuracy: defines whether to report the prediction or not
        """

        if self.model_name:
            from pyfasttext import FastText
            predictor = FastText()
            predictor.load_model('ft_extras/'+self.model_name+'.bin')
            predicted_labels = predictor.predict_proba(test_set)
            if report_accuracy and test_labels_vector:
                test_set_size = len(test_set)
                correct_predictions = 0
                invalid_labels = 0
                for index, labels in enumerate(predicted_labels):
                    if len(labels) != 0:
                        best_label = max(labels,key=lambda label:label[1])
                        if best_label[0] == test_labels_vector[index]:
                            correct_predictions += 1
                    else:
                        invalid_labels += 1
                        continue
                print('Prediction accuracy:{}\n'.format(correct_predictions / (test_set_size - invalid_labels)))
        else:
            print('Please use the train method to train a model first.')
            return

if __name__ == '__main__':
    emotions = ['joy', 'fear', 'anger', 'sadness']
    classifier = Classifier()
    from classifier.preclassifier import fetch_data
    for emotion in emotions:
        print('Predicting for {}:'.format(emotion))
        train_tweets, test_tweets, train_labels, test_labels = fetch_data(emotion,vectorize=False)
        classifier.train(train_tweets,train_labels)
        classifier.predict(test_tweets,test_labels)