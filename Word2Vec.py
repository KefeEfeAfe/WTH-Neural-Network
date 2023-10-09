import numpy as np
import tensorflow as tf

# Training data
corpus = [
    "the dog saw a cat",
    "the dog chased the cat",
    "the cat climbed a tree"
]

# Preprocessing the data and creating the dictionary
words = []
for sentence in corpus:
    words.extend(sentence.split())
words = list(set(words))
word_to_index = {word: i for i, word in enumerate(words)}
vocab_size = len(words)


# Function to create one-hot vectors
def create_one_hot_vector(word):
    vector = np.zeros(vocab_size)
    vector[word_to_index[word]] = 1
    return vector


# Neural network parameters
input_size = vocab_size
hidden_size = 3
output_size = vocab_size
learning_rate = 0.01
epochs = 1000

# Initialize weights
np.random.seed(0)
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)

# Training the neural network
for epoch in range(epochs):
    total_loss = 0

    # Iterate through the corpus
    for sentence in corpus:
        target_word = "climbed"
        input_vector = create_one_hot_vector(target_word)

        # Forward pass
        hidden_activation = np.dot(input_vector, weights_input_hidden)
        hidden_output = 1 / (1 + np.exp(-hidden_activation))
        output_activation = np.dot(hidden_output, weights_hidden_output)
        output_probs = np.exp(output_activation) / np.sum(np.exp(output_activation))

        # Calculate cross-entropy loss
        target_index = word_to_index[target_word]
        loss = -np.log(output_probs[target_index])
        total_loss += loss

        # Backpropagation

        # Output layer gradient
        output_grad = output_probs
        output_grad[target_index] -= 1

        # Hidden layer gradient
        hidden_grad = np.dot(output_grad, weights_hidden_output.T) * hidden_output * (1 - hidden_output)

        # Update weights using gradient descent
        weights_hidden_output -= learning_rate * np.outer(hidden_output, output_grad)
        weights_input_hidden -= learning_rate * np.outer(input_vector, hidden_grad)

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(corpus)}')

# Use the trained network to predict probabilities for the word "cat"
input_vector = create_one_hot_vector("cat")
hidden_output = 1 / (1 + np.exp(-np.dot(input_vector, weights_input_hidden)))
output_probs = np.exp(np.dot(hidden_output, weights_hidden_output)) / np.sum(
    np.exp(np.dot(hidden_output, weights_hidden_output)))

# Display the probabilities for all words in the dictionary
for word, prob in zip(words, output_probs):
    print(f'Probability for "{word}": {prob:.4f}')
