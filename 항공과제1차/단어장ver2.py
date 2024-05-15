with open('processed_data.txt', 'r') as f:
    sequences = [line.strip() for line in f]
    
subsequences = set(sequences)

with open('단어장ver2.txt', 'w') as f:
    # 데이터를 파일에 저장
    for line in subsequences:
        f.write(line + '\n')