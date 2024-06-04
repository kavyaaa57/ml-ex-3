import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(self.num_layers-1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(self.num_layers-1)]
    
    # Forward pass
    def forward(self, X):
        self.activations = [X]
        self.layer_inputs = []
        for i in range(self.num_layers - 1):
            layer_input = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.layer_inputs.append(layer_input)
            activation = sigmoid(layer_input)
            self.activations.append(activation)
        return self.activations[-1]
    
    # Backward pass
    def backward(self, X, y, output, learning_rate):
        errors = [y - output]
        for i in range(self.num_layers - 2, -1, -1):
            delta = errors[-1] * sigmoid_derivative(self.activations[i+1])
            errors.append(delta.dot(self.weights[i].T))
            self.weights[i] += self.activations[i].T.dot(delta) * learning_rate
            self.biases[i] += np.sum(delta, axis=0, keepdims=True) * learning_rate

# Define the main Streamlit app
def main():
    st.title('Backpropagation Demo')

    # User-defined parameters
    num_layers = st.slider('Number of Layers', min_value=2, max_value=5, value=3)
    layer_sizes = [st.slider(f'Layer {i+1} Size', min_value=1, max_value=10, value=4) for i in range(num_layers)]
    learning_rate = st.slider('Learning Rate', min_value=0.01, max_value=1.0, value=0.1)
    epochs = st.slider('Number of Epochs', min_value=100, max_value=1000, value=500)

    # Create the neural network model
    nn = NeuralNetwork(layer_sizes)

    # Generate random training data
    np.random.seed(42)
    X = np.random.randn(100, layer_sizes[0])
    y = np.random.randint(2, size=(100, layer_sizes[-1]))

    # Train the neural network using backpropagation
    losses = []
    for epoch in range(epochs):
        output = nn.forward(X)
        loss = np.mean(np.square(y - output))
        nn.backward(X, y, output, learning_rate)
        losses.append(loss)

    # Plot the loss curve
    st.subheader('Loss Curve')
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    st.pyplot(fig)

    # Display final weights and biases
    st.subheader('Final Weights and Biases')
    for i in range(num_layers - 1):
        st.write(f'Layer {i+1} - Weights:')
        st.write(nn.weights[i])
        st.write(f'Layer {i+1} - Biases:')
        st.write(nn.biases[i])

if __name__ == "__main__":
    main()
