import pandas as pd

# 파일 이름을 설정해주세요
file1 = '2019_data.csv'
file2 = '2022_data.csv'

# 파일들을 읽어서 DataFrame으로 변환
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# event와 event_time 열만 선택
df1_selected = df1[['event', 'event_time']]
df2_selected = df2[['event', 'event_time']]

# 두 DataFrame을 합치기
df = pd.concat([df1_selected, df2_selected], ignore_index=True)

# 결과를 새로운 CSV 파일에 저장
df.to_csv('test_data.csv', index=False)
