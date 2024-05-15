import pandas as pd

# 파일 이름을 설정해주세요
file1 = '2019_data.csv'
file2 = '2022_data.csv'

# 파일들을 읽어서 DataFrame으로 변환
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# event와 event_time 열만 선택
df1_data = df1[['event', 'event_time']]
df2_data = df2[['event', 'event_time']]

# 두 개의 데이터프레임을 합침
df = pd.concat([df1_data, df2_data], ignore_index=True)

# 결과를 새로운 CSV 파일에 저장
df.to_csv('data.csv', index=False)
