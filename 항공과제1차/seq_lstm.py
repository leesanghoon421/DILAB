import numpy as np
import tensorflow as tf
import pandas as pd
import math
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle
from keras.layers import Dropout
from keras.layers import GRU, Bidirectional

with open('processed_data.txt', 'r') as f:
    sequences = [line.strip() for line in f]

def generate_subsequences(sequences):
    subsequences = set()
    for sequence in sequences:
        tokens = sequence.split()
        for i in range(1, len(tokens)+1):
            subsequences.add('/'.join(tokens[:i]))
    return list(set(subsequences))

# 모든 부분 시퀀스를 생성
subsequences = generate_subsequences(sequences)
subsequences = [seq + ' <END>' for seq in subsequences]

# 빈도수를 계산
sequence_counts = {}
for sequence in sequences:
    if sequence in sequence_counts:
        sequence_counts[sequence] += 1
    else:
        sequence_counts[sequence] = 1

# 빈도수를 기준으로 시퀀스를 정렬
sorted_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)

# 빈도수가 너무 높은 시퀀스의 수를 줄임
new_sequences = []
for sequence, count in sorted_sequences:
    if count > 10000:
        count = math.floor(count * 0.6)
    if count > 5000:
        count = math.floor(count * 0.6)
    if count > 2000:
        count = math.floor(count * 0.6)
    if count > 1500:
        count = math.floor(count * 0.6)
    if count > 1300:
        count = math.floor(count * 0.6)
    if count > 1100:
        count = math.floor(count * 0.6)
    if count > 800:
        count = math.floor(count * 0.7)
    if count > 400:
        count = math.floor(count * 0.7)
    if count > 200:
        count = math.floor(count * 0.7)
    if count > 100:
        count = math.floor(count * 0.7)
    new_sequences += [sequence] * count

# 토큰화
tokenizer = Tokenizer(filters='\t\n', lower=False)
tokenizer.fit_on_texts([seq + ' <END>' for seq in new_sequences])

# Create a set of all individual tokens in the training data
training_tokens = set(tokenizer.word_index.keys())

# Save the set of training tokens
with open('training_tokens.pickle', 'wb') as handle:
    pickle.dump(training_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)


total_words = len(tokenizer.word_index) + 1

# Save the tokenizer

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 부분 시퀀스를 사용하여 학습 데이터를 생성
input_sequences = []
for line in new_sequences:
    token_list = tokenizer.texts_to_sequences([line + ' <END>'])[0]
    for i in range(1, len(token_list)):
        temp_sequence = token_list[:i+1]
        input_sequences.append(temp_sequence)

# 패딩
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

print(tokenizer.word_index)

xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(256, return_sequences=True))  # 첫 번째 LSTM 레이어의 유닛 수를 256으로 변경
model.add(Dropout(0.2))  # Dropout 레이어 추가
model.add(LSTM(128))  # 두 번째 LSTM 레이어 추가
model.add(Dense(total_words, activation='softmax'))
adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# 학습 데이터와 검증 데이터를 분할합니다. 검증 데이터는 전체의 20%를 사용합니다.
xs_train, xs_val, ys_train, ys_val = train_test_split(xs, ys, test_size=0.2, random_state=42)

# 모델 학습
history = model.fit(xs_train, ys_train, epochs=20, verbose=1, validation_data=(xs_val, ys_val))

# 그래프로 학습 과정 시각화
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

# 모델 저장
model.save('my_model.h5')

# Save new_sequences
with open('new_sequences.pickle', 'wb') as handle:
    pickle.dump(new_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
