with open('processed_data.txt', 'r') as f:
    sequences = [line.strip() for line in f]

subsequences = []
for sequence in sequences:
    tokens = sequence.split()
    for i in range(1, len(tokens)+1):
        subsequence = ' '.join(tokens[:i])
        subsequences.append(subsequence)
subsequences = set(subsequences)
# 파일 생성
with open('단어장.txt', 'w') as f:
    # 데이터를 파일에 저장
    for line in subsequences:
        f.write(line + '\n')