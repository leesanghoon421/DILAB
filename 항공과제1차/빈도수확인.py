import pandas as pd

with open('processed_data.txt', 'r') as f:
    sequences = [line.strip() for line in f]

# 빈도수 계산
sequence_counts = {}
for sequence in sequences:
    split_sequence = sequence.split()
    for item in split_sequence:
        if item in sequence_counts:
            sequence_counts[item] += 1
        else:
            sequence_counts[item] = 1

# 빈도수를 기준으로 시퀀스를 정렬
sorted_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)

# 빈도수 출력
for sequence, count in sorted_sequences:
    print(f"{sequence}\t{count}")
