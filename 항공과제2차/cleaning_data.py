import pandas as pd

# 파일 이름을 설정해주세요
file1 = 'processed_data.csv'

# 파일을 읽어서 DataFrame으로 변환
df = pd.read_csv(file1)

# 반복되는 시퀀스 제거
for idx, row in df.iterrows():
    # 이전 event와 event_time의 초기값 설정
    prev_event = None
    events = row['event']
    events_time = row['event_time']
    seq_list = events.split(' | ')
    time_seq_list = events_time.split(' | ')
    
    # 제거할 인덱스 저장
    remove_indexes = []

    for i in range(len(seq_list)):
        event = seq_list[i]
        event_time = time_seq_list[i]

        if prev_event is None:
            prev_event = event
        elif i == (len(seq_list)-1):
            continue
        elif prev_event == event:
            remove_indexes.append(i)
        else:
            if remove_indexes and seq_list[i-1]==seq_list[i-2]:
                remove_indexes.pop()
            prev_event = event

    # 내부 리스트를 수정하기 위해 복사하여 사용
    modified_list = seq_list.copy()
    modified_time_list = time_seq_list.copy()

    for index in sorted(remove_indexes, reverse=True):
        del modified_list[index]
        del modified_time_list[index]

    # 형식 맞춰서 DataFrame에 저장
    df.at[idx, 'event'] = ' | '.join(modified_list)
    df.at[idx, 'event_time'] = ' | '.join(modified_time_list)

# 수정된 DataFrame을 새로운 파일로 저장
df.to_csv('cleaned_data.csv', index=False)