import pandas as pd
import math

with open('processed_data.txt', 'r') as f:
    sequences = [line.strip() for line in f]

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

# Pandas Series로 변환
data_series = pd.Series(new_sequences)

# 빈도수 계산
frequency = data_series.value_counts()

# 빈도수 상위 10개 출력
print(frequency.head(30))
