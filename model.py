from layers import Dense, LeakyReLU, Softmax, CrossEntropy
import numpy as np
import pickle


class NeuralNet:
    """
    Created architecture to solve given task
    """
    def __init__(self, in_dim, random_state=2020):
        """
        Initiation model. Here is needed to define model layer.
        :param in_dim: Input dimension for first layer
        :param random_state: To fix random generation in weights.
        """
        self.seed = random_state  # Fix this seed
        np.random.seed(seed=self.seed)
        self.Dense1 = Dense(in_dim, 64)  # Create first dense layer
        self.LeakyReLU1 = LeakyReLU(0.0003)  # Create activation layer
        self.Dense2 = Dense(64, 10)  # Create second dense layer
        self.Softmax = Softmax()  # Softmax layer
        self.criterion = CrossEntropy()  # Loss function

        self.history = []  # List to save loss across epoches
        self.accuracy_valid = []  # List to save validation accuracy across epoches
        self.accuracy_train = []  # List to save train accuracy across epoches
        self.max_acc = 0  # Save maximum reached accuracy

        self.out = None
        self.flow = None

    def forward_step(self, X, y_ohe, mode='train'):
        """
        Process given data through model. Here is defined forward sequence of layer.
        Function can be used to training process and for inference
        :param X: Given data
        :param y_ohe: Ground truth labels
        :param mode: Mode for forward processing. Can be 'train', 'eval'
        :return:
        """
        np.random.seed(seed=self.seed)
        if mode == 'train':
            output = self.Dense1.forward(X)
            output = self.LeakyReLU1.forward(output)
            output = self.Dense2.forward(output)
            output = self.Softmax.forward(output)
            loss = self.criterion.forward(output, y_ohe)
            self.history.append(loss)
            return output
        else:
            # Load scaler parameters to scale evaluation data
            with open('scaler_params.pickle', 'rb') as file:
                params = pickle.load(file)
            X = (X - params['means']) / (params['std'] + 1e-8)  # Scale data
            output = self.Dense1.forward(X)
            output = self.LeakyReLU1.forward(output)
            output = self.Dense2.forward(output)
            output = self.Softmax.forward(output)
            prediction = np.argmax(output, axis=1)  # Take max probability
            return prediction

    def backward_step(self, pred, y_ohe, lr, reg):
        """
        Make backpropagation through all layer.
        :param pred: Prediction from neural net
        :param y_ohe: Ground truth labels
        :param lr: Learning rate for gradient descent
        :param reg: L2 regularization coefficient
        :return:
        """
        np.random.seed(seed=self.seed)
        flow = self.criterion._backward(pred, y_ohe)
        flow = self.Dense2._backward(flow, lr, reg)
        flow = self.LeakyReLU1._backward(flow)
        flow = self.Dense1._backward(flow, lr, reg)
        return

    def validate_step(self, X, y_ohe, mode):
        """
        Calculate accuracy on model
        :param X: Data to process
        :param y_ohe: Ground truth labels
        :param mode: Mode to validation. Can be 'train', 'val'.
        :return: Current accuracy
        """
        np.random.seed(seed=self.seed)
        val = self.forward_step(X, y_ohe)  # Answers from model
        predict_val = np.argmax(val, axis=1)  # Receive class
        accuracy = (predict_val == np.argmax(y_ohe, axis=1)).sum() / len(predict_val)  # Calculate accuracy
        if mode == 'val':
            self.accuracy_valid.append(accuracy)  # Save to accuracy_valid list
        else:
            self.accuracy_train.append(accuracy)  # Save to accuracy_train list
        return accuracy

    def check_and_save(self, acc):
        """
        Save model weights with best accuracy
        :param acc: Current accuracy
        :return:
        """
        if acc > self.max_acc:  # If current accuracy is bigger than max_acc
            self.max_acc = acc  # Update max_acc
            self.save_weights()  # Save model weights
        return

    def save_weights(self):
        """
        Save weights of model in weights_model.pickle file
        :return:
        """
        weights = {'Dense1': self.Dense1.W,
                   'Dense2': self.Dense2.W}  # Define dict to future easy access to data

        # Save weights
        with open('wights_model.pickle', 'wb') as file:
            pickle.dump(weights, file, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def load_weights(self):
        """
        Load weights from folder with model.py file and apply it to model.
        Name of file with weights must me wights_model.pickle.
        :return:
        """
        # Load file
        with open('wights_model.pickle', 'rb') as file:
            weights = pickle.load(file)
        self.Dense1.W = weights['Dense1']  # Set weights to first dense layer
        self.Dense2.W = weights['Dense2']  # Set weights to second dense layer
        return

