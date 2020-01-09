# Shanker, Avinash
# 1001-668-570
# 2019-10-25
# Assignment-03-01

# using tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each the input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None

    def add_layer(self, num_nodes, activation_function):
        if self.weights:
            layer_weights = np.array([[np.random.rand() for i in range(num_nodes)] for j in range(self.weights[-1].shape[1])])
        else:
            layer_weights = np.array([[np.random.normal() for i in range(num_nodes)] for j in range(self.input_dimension)])
        layer_bias = np.array([[np.random.rand() for column in range(num_nodes)]])
        
        self.weights.append(tf.Variable(layer_weights, trainable=True))
        self.biases.append(tf.Variable(layer_bias, trainable=True))
        self.activations.append(activation_function)
    
    def get_weights_without_biases(self, layer_number):
        return self.weights[layer_number]
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """

    def get_biases(self, layer_number):
        return self.biases[layer_number]
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases). Note that the biases shape should be [1][number_of_nodes]
         """

    def set_weights_without_biases(self, weights, layer_number):
        self.weights[layer_number] = weights
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """

    def set_biases(self, biases, layer_number):
        self.biases[layer_number] = biases
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """

    def set_loss_function(self, loss_fn):
        """
        This function sets the loss function.
        :param loss_fn: Loss function
        :return: none
        """
        self.loss = loss_fn

    def sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def linear(self, x):
        return x

    def relu(self, x):
        out = tf.nn.relu(x)
        return out

    def cross_entropy_loss(self, y, y_hat):
        """
        This function calculates the cross entropy loss
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual outputs values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        tf_inputX = tf.Variable(X)
        weight_len = len(self.weights)
        for network in range(weight_len):
            tf_weight = tf.matmul(tf_inputX, self.get_weights_without_biases(network))
            tf_bias = tf.add(tf_weight, self.get_biases(network))
            tf_inputX = self.activations[network](tf_bias)
        return tf_inputX
            

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8, regularization_coeff=1e-6):
         """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :param regularization_coeff: regularization coefficient
         :return: None
         """
         input_shape = X_train.shape[0]
         for num in range(num_epochs):
             for set in range(0, input_shape, batch_size):
                 rows = set + batch_size
                 if rows > input_shape:
                    rows = input_shape
                 tf_x_train = tf.Variable(X_train[set : rows,:])
                 tf_y_train = tf.Variable(y_train[set : rows])
                 
                 with tf.GradientTape(persistent=True) as tape:
                     tape.watch(self.biases)
                     tape.watch(self.weights)
                     prediction = self.predict(tf_x_train)
                     CEL_loss_fun = self.cross_entropy_loss(tf_y_train, prediction)
                 
                 weight_len = len(self.weights)
                 for tier in range(weight_len):
                     dloss_dweight = tape.gradient(CEL_loss_fun, self.get_weights_without_biases(tier))
                     dloss_dbias = tape.gradient(CEL_loss_fun, self.get_biases(tier))

                     w_new = self.get_weights_without_biases(tier) - (alpha * dloss_dweight)
                     b_new = self.get_biases(tier) - (alpha * dloss_dbias)
                     self.set_weights_without_biases(w_new, tier)
                     self.set_biases(b_new, tier)

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        prediction = self.predict(X)
        prediction = prediction.numpy()
        row = y.shape[0]
        col = self.weights[-1].shape[1]
        Zero_Temp = np.zeros((col,row))
        Zero_Temp[y, np.arange(y.shape[0])] = 1
        temp_e = Zero_Temp.T
        max = prediction.argmax(axis=1)
        output_shape = prediction.shape[0]
        temp_r = (max[:, None] == np.arange(prediction.shape[1])).astype(float)
        err = 0
        for sample in range(output_shape):
            if not np.allclose(temp_e[sample], temp_r[sample]):
                err = err + 1
        return err / output_shape

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """
        n = self.weights[-1].shape[1]
        m = y.shape[0]
        conf_ma = np.zeros((n, n))
        prediction = self.predict(X)
        prediction = prediction.numpy()
        Zero_Temp = np.zeros((n,m))
        Zero_Temp[y, np.arange(y.shape[0])] = 1
        temp_e = Zero_Temp.T
        maxi = prediction.argmax(axis=1)
        temp_r = (maxi[:, None] == np.arange(prediction.shape[1])).astype(float)
        for i in range(temp_r.shape[0]):
            coordinates = np.where(temp_r[i] == 1)
            c_size = coordinates[0].size
            if c_size != 0:
                actual_class = coordinates[0][0]
                if np.array_equal(temp_r[i], temp_e[i]):
                    conf_ma[actual_class, actual_class] += 1
                else:
                    conf_ma[y[0], actual_class] += 1                   
        return conf_ma


if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist

    np.random.seed(seed=1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape and Normalize data
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]
    indices = list(range(X_train.shape[0]))
    # np.random.shuffle(indices)
    number_of_samples_to_use = 500
    X_train = X_train[indices[:number_of_samples_to_use]]
    y_train = y_train[indices[:number_of_samples_to_use]]
    multi_nn = MultiNN(input_dimension)
    number_of_classes = 10
    activations_list = [multi_nn.sigmoid, multi_nn.sigmoid, multi_nn.linear]
    number_of_neurons_list = [50, 20, number_of_classes]
    for layer_number in range(len(activations_list)):
        multi_nn.add_layer(number_of_neurons_list[layer_number], activation_function=activations_list[layer_number])
    for layer_number in range(len(multi_nn.weights)):
        W = multi_nn.get_weights_without_biases(layer_number)
        W = tf.Variable((np.random.randn(*W.shape)) * 0.1, trainable=True)
        multi_nn.set_weights_without_biases(W, layer_number)
        b = multi_nn.get_biases(layer_number=layer_number)
        b = tf.Variable(np.zeros(b.shape) * 0, trainable=True)
        multi_nn.set_biases(b, layer_number)
    multi_nn.set_loss_function(multi_nn.cross_entropy_loss)
    percent_error = []
    for k in range(10):
        multi_nn.train(X_train, y_train, batch_size=100, num_epochs=20, alpha=0.8)
        percent_error.append(multi_nn.calculate_percent_error(X_train, y_train))
    confusion_matrix = multi_nn.calculate_confusion_matrix(X_train, y_train)
    print("Percent error: ", np.array2string(np.array(percent_error), separator=","))
    print("************* Confusion Matrix ***************\n", np.array2string(confusion_matrix, separator=","))
