import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
import pickle

# Load the saved model
model = load_model('my_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the max sequence length based on the training data
max_sequence_len = model.layers[0].input_shape[1]

# Load training tokens
with open('training_tokens.pickle', 'rb') as handle:
    training_tokens = pickle.load(handle)

def predict_next_sequence(input_sequence, num_predictions):
    predicted_tokens = []
    seed_text = input_sequence + " <END>"
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]

    # Get the indices of the top 'num_predictions' predictions
    top_indices = predicted_probs.argsort()[-num_predictions:][::-1]
    top_probs = predicted_probs[top_indices]
    total_prob = np.sum(top_probs)

    # Normalize probabilities to sum up to 1
    normalized_probs = top_probs / total_prob

    for i, predicted in enumerate(top_indices):
        output_token = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted and word in training_tokens:
                output_token = word
                break
        if output_token and i >= 2:  # 3등부터 출력하도록 수정
            predicted_tokens.append((output_token, normalized_probs[i]))

    # Adjust probabilities to sum up to 1
    sum_probs = np.sum([prob for _, prob in predicted_tokens])
    normalized_tokens = [(token, prob/sum_probs) for token, prob in predicted_tokens]

    return normalized_tokens

input_sequences = ['HE']
num_predictions = 7

for input_sequence in input_sequences:
    print("Input Sequence:", input_sequence)
    print("Predicted Sequences:")
    predicted_tokens = predict_next_sequence(input_sequence, num_predictions)
    for token, probability in predicted_tokens:
        print(token, "(Probability:", probability, ")")
    print()
