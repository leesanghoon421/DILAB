import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.optimizers import Adam

with open('processed_data.txt', 'r') as f:
    sequences = [line.strip() for line in f]

def generate_subsequences(sequences):
    subsequences = set()
    for sequence in sequences:
        tokens = sequence.split()
        for i in range(1, len(tokens)+1):
            # 공백을 /로 대체
            subsequences.add('/'.join(tokens[:i]))
    return list(set(subsequences))

# 모든 부분 시퀀스를 생성
subsequences = generate_subsequences(sequences)
subsequences = [seq + ' <END>' for seq in subsequences]

# 토큰화
tokenizer = Tokenizer(filters='\t\n', lower=False)
tokenizer.fit_on_texts(subsequences)

total_words = len(tokenizer.word_index) + 1

# 원본 시퀀스를 사용하여 학습 데이터를 생성
input_sequences = []
for line in sequences:
    token_list = tokenizer.texts_to_sequences([line + ' <END>'])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# 패딩
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

print(tokenizer.word_index)
print(max([len(x) for x in input_sequences]))