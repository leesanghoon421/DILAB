import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras import backend as K

# Initialize tokenizer with the given settings
tokenizer = Tokenizer(num_words=None, oov_token=None, filters='', lower=False, split='|')

# Use your sequences
with open('data.csv', 'r') as f:
    raw_sequences = [line.strip().split(' | ') for line in f]

# Add <END> token to the end of each sequence
raw_sequences = [seq + ["<END>"] for seq in raw_sequences]

# Fit tokenizer on texts
tokenizer.fit_on_texts(raw_sequences)

# Convert text to sequence of integers
sequences = tokenizer.texts_to_sequences(raw_sequences)

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=14, padding='pre')

# Prepare targets
targets = np.roll(padded_sequences, shift=-1, axis=1)

# Calculate num_classes
num_classes = max([max(seq) for seq in targets]) + 1

# Convert targets to categorical format
categorical_targets = to_categorical(targets, num_classes=num_classes)

# Split data into train and validation sets
padded_sequences_train, padded_sequences_val, categorical_targets_train, categorical_targets_val = train_test_split(padded_sequences, categorical_targets, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=num_classes, output_dim=64, input_length=14),
    LSTM(256, return_sequences=True),
    Dropout(0.1),
    Dense(num_classes, activation='softmax')
])

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

# Compile model
model.compile(loss=weighted_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

# Display the model architecture
model.summary()

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(padded_sequences_train, categorical_targets_train, epochs=100, validation_data=(padded_sequences_val, categorical_targets_val), callbacks=[early_stopping])

# Plot the training history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save the tokenizer
with open('tokenizer_last.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the model
model.save('my_model_last.h5')