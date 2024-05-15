from keras.models import load_model
from keras.utils import pad_sequences
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

def weighted_categorical_crossentropy(y_true, y_pred):
    end_token_id = tokenizer.word_index['<END>']
    ofa_class_index = tokenizer.word_index['OFA']
    end_token_weight = 0.0001  # end_token weight control
    ofa_weight = 0.1  # OFA weight control

    # Get the true classes
    y_true_classes = K.argmax(y_true, axis=-1)

    # Create a tensor for the weights
    weights = tf.where(tf.equal(y_true_classes, end_token_id), end_token_weight, 1.0)
    weights = tf.where(tf.equal(y_true_classes, ofa_class_index), ofa_weight, weights)

    # Calculate the unweighted loss
    unweighted_loss = K.categorical_crossentropy(y_true, y_pred)

    # Apply the weights
    weighted_loss = unweighted_loss * weights

    return K.mean(weighted_loss)


# Load the tokenizer
with open('tokenizer_last.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = load_model('my_model_last.h5', custom_objects={'weighted_categorical_crossentropy': weighted_categorical_crossentropy})


def seq_prediction(input_text, graph, parent, max_len=14, depth=0, max_depth=3):
    if depth > max_depth:
        return

    encoded = tokenizer.texts_to_sequences([input_text])[0]
    padded_sequence = pad_sequences([encoded], maxlen=max_len, padding='pre')
    preds = model.predict(padded_sequence)[0]

    top3_predicted_ids = np.argsort(preds[-1])[-3:][::-1]
    top3_predicted_words = [tokenizer.index_word.get(idx, "<UNK>") for idx in top3_predicted_ids]
    top3_predicted_probs = preds[-1][top3_predicted_ids]

    for word, prob in zip(top3_predicted_words, top3_predicted_probs):
        new_input_text = input_text + ' ' + word
        node_name = f'{new_input_text} ({prob:.2f})'
        graph.add_edge(parent, node_name)

        if word != '<END>':
            seq_prediction(new_input_text, graph, node_name, max_len=max_len, depth=depth+1, max_depth=max_depth)

def seq_prediction2(input_text, max_len=14):
    encoded = tokenizer.texts_to_sequences([input_text])[0]
    padded_sequence = pad_sequences([encoded], maxlen=max_len, padding='pre')
    preds = model.predict(padded_sequence)[0]

    top5_predicted_ids = np.argsort(preds[-1])[-5:][::-1]
    top5_predicted_words = [tokenizer.index_word.get(idx, "<UNK>") for idx in top5_predicted_ids]
    top5_predicted_probs = preds[-1][top5_predicted_ids]

    print(f'Input: {input_text}')
    for word, prob in zip(top5_predicted_words, top5_predicted_probs):
        print(f'Predicted word: {word}, Probability: {prob:.2f}')

seq_prediction2('GAA')

# Initialize a directed graph
G = nx.DiGraph()

# Generate text
seq_prediction('GAA', G, 'GAA')

# Draw the graph
plt.figure(figsize=(20, 15)) # you can change the size as you need

pos = nx.spring_layout(G, k=0.3)  # using graphviz layout for better visualization
nx.draw(G, pos, with_labels=True, arrows=True)

plt.show()