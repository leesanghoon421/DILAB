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
    
# Load new_sequences
with open('new_sequences.pickle', 'rb') as handle:
    new_sequences = pickle.load(handle)


# Define the max sequence length based on the training data
max_sequence_len = model.layers[0].input_shape[1]  # 첫 번째 레이어의 입력 형상을 기반으로 최대 시퀀스 길이를 정의합니다.

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

    for predicted in top_indices:
        output_token = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted and word in training_tokens:  # Check if the predicted word is in the training data
                output_token = word
                break
        if output_token:  # If a valid word is found, add it to the output
            predicted_tokens.append((output_token, predicted_probs[predicted]))

    return predicted_tokens

input_sequences = ['HE']
num_predictions = 5

for input_sequence in input_sequences:
    print("Input Sequence:", input_sequence)
    print("Predicted Sequences:")
    predicted_tokens = predict_next_sequence(input_sequence, num_predictions)
    for token, probability in predicted_tokens:
        print(token, "(Probability:", probability, ")")
    print()