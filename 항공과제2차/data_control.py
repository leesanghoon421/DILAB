import pandas as pd

# 데이터 파일 로드
data = pd.read_csv('data.csv')

# event와 event_time을 쌍으로 묶은 딕셔너리 생성
event_dict = {}
for idx, row in data.iterrows():
    events = row['event'].split(' | ')
    event_time = row['event_time']
    for event in events:
        if event in event_dict:
            event_dict[event].append(event_time)
        else:
            event_dict[event] = [event_time]

# event_time이 같은 event를 '='으로 연결하고, 해당 event_time은 하나만 남기는 함수
def join_events(events):
    if len(events) == 1:
        return events[0]
    return ' | '.join([f'{event}={event_time}' for event, event_time in zip(events, set(events))])

# 새로운 event_time 값을 생성한 후 데이터프레임에 추가
data['event'] = data['event'].apply(lambda x: ' | '.join(x.split(' | ')))
data['event_time'] = data['event'].map(event_dict).apply(join_events)

# 중복된 행 제거 (첫 번째 행만 남김)
data = data.drop_duplicates(subset='event_time', keep='first')

# 결과를 새로운 CSV 파일로 저장
data.to_csv('control_data.csv', index=False)
