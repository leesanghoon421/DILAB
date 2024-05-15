import pandas as pd

# 파일 이름을 설정해주세요
file1 = 'test_data.csv'

# 파일을 읽어서 DataFrame으로 변환
df = pd.read_csv(file1)

def merge(lst, indices):
    result = []
    merged_indices = []
    for idx in indices:
        if idx + 1 in indices and idx not in merged_indices:
            result.append(lst[idx] + "=" + lst[idx + 1])
            merged_indices.extend([idx, idx + 1])
        elif idx not in merged_indices:
            result.append(lst[idx])
        else:
            result.append(lst[idx])
    return result

def merge_time(lst, indices):
    result = []
    merged_indices = []
    for idx in indices:
        if idx + 1 in indices and idx not in merged_indices:
            result.append(lst[idx])
            merged_indices.extend([idx, idx + 1])
        elif idx not in merged_indices:
            result.append(lst[idx])
        else:
            result.append(lst[idx])
    return result

# 반복되는 시퀀스 제거
for idx, row in df.iterrows():
    # 이전 event와 event_time의 초기값 설정
    prev_time = None
    events = row['event']
    events_time = row['event_time']
    seq_list = events.split(' | ')
    time_seq_list = events_time.split(' | ')

    joint_indexes = []
    
    for i in range(len(seq_list)):
        event = seq_list[i]
        event_time = time_seq_list[i]
        if prev_time is None:
            prev_time = event_time
        elif prev_time == event_time:
            if i-2 in joint_indexes:
                joint_indexes.append(i)
            else:
                joint_indexes.append(i)
                joint_indexes.append(i-1)
        else:
            prev_time = event_time

    # 내부 리스트를 수정하기 위해 복사하여 사용
    modified_list = seq_list.copy()
    modified_time_list = time_seq_list.copy()

    for index in sorted(joint_indexes, reverse=True):
        modified_list = merge(seq_list,joint_indexes)
        modified_time_list = merge_time(time_seq_list,joint_indexes)

    # 형식 맞춰서 DataFrame에 저장
    df.at[idx, 'event'] = ' | '.join(modified_list)
    df.at[idx, 'event_time'] = ' | '.join(modified_time_list)

# 수정된 DataFrame을 새로운 파일로 저장
df.to_csv('processed_data.csv', index=False)