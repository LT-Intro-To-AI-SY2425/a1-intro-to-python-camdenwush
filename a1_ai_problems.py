import numpy as np

# idk how to do this better
word_to_num = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}

def encode_word(word, max_len=5):
    """
    encodes a word into a fixedsize one hot encoded vector because the nn
    expects a number, also you need same number of features
    """
    
    # how do u do this better
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    encoding = np.zeros((max_len, len(alphabet)))
    for i, char in enumerate(word):
        encoding[i, alphabet.index(char)] = 1
    return encoding.flatten()

X = np.array([encode_word(word) for word in word_to_num.keys()])
y = np.array([word_to_num[word] for word in word_to_num.keys()]).reshape(-1, 1)

class Stupid:
    """
    one hidden layer, sigmoid
    """
    
    def __init__(self):
        """
        setup random weights and set the rate
        """
        np.random.seed(1)
        # 10 hidden neurons
        self.weights1 = np.random.rand(X.shape[1], 10)
        self.weights2 = np.random.rand(10, 1)
        self.lr = 0.01
    
    def sigmoid(self, x):
        #sigma
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        derivative of the sigmoid function is needed for backpropagation 
        """
        return x * (1 - x)
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights1)
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_output = np.dot(self.hidden_output, self.weights2)
        return self.final_output
    
    def backward(self, X, y, output):
        """
        actually do the backpropagation based off error
        """
        error = y - output
        d_weights2 = np.dot(self.hidden_output.T, error)
        d_weights1 = np.dot(X.T, (np.dot(error, self.weights2.T) * self.sigmoid_derivative(self.hidden_output)))

        self.weights1 += self.lr * d_weights1
        self.weights2 += self.lr * d_weights2
    
    def train(self, X, y, iterations=1000):
        for i in range(iterations):
            output = self.forward(X)
            self.backward(X, y, output)
            if (i+1) % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"i={i+1}, loss;{loss}")

nn = Stupid()
nn.train(X, y)

for word in word_to_num.keys():
    test_input = encode_word(word).reshape(1, -1)
    prediction = nn.forward(test_input)
    print(f"i think {word} is:", round(prediction[0][0]))
