import pandas as pd
import numpy as np
from datetime import datetime

# CSV 파일에서 데이터 읽어오기
df = pd.read_csv('./항공과제2차/data.csv')

# 시간대 분류 함수
def classify_time(hour):
    if 0 <= hour < 6:
        return 'Dawn'
    elif 6 <= hour < 12:
        return 'A.M'
    elif 12 <= hour < 18:
        return 'P.M'
    else:
        return 'Night'

# 데이터 처리를 위한 리스트 선언
months = []
time_of_day = []

# 데이터 처리
for idx, row in df.iterrows():
    event_times = row['event_time'].split(' | ')
    
    # 예외 처리: 형식에 맞지 않는 값은 건너뛰기
    if len(event_times) < 2:
        continue
    
    try:
        start_time = datetime.strptime(event_times[0], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(event_times[-1], '%Y-%m-%d %H:%M:%S')
    except ValueError:
        continue
    
    # 월 추출
    months.append(start_time.month)
    
    # 시작 시간대 추출
    start_time_of_day = classify_time(start_time.hour)
    time_of_day.append(start_time_of_day)

# 새로운 데이터 프레임 생성
new_df = pd.DataFrame({'Month': months, 'TimeOfDay': time_of_day})

# 월별, 시간대별 사건의 비율 계산
month_ratio = new_df['Month'].value_counts(normalize=True).sort_index()
time_of_day_ratio = new_df['TimeOfDay'].value_counts(normalize=True).reindex(['Dawn', 'A.M', 'P.M', 'Night'])

print("월별 사건 비율:")
print(month_ratio)
print("\n시간대별 사건 비율:")
print(time_of_day_ratio)


import matplotlib.pyplot as plt

# 월별 사건 비율 시각화
plt.figure(figsize=(10, 5))
plt.bar(month_ratio.index, month_ratio.values)
plt.xlabel('month')
plt.ylabel('ratio')
plt.title('Monthly event ratio')
plt.show()

# 시간대별 사건 비율 시각화
plt.figure(figsize=(8, 6))
plt.pie(time_of_day_ratio.values, labels=time_of_day_ratio.index, autopct='%1.1f%%', startangle=90)
plt.title('time of day ratio')
plt.axis('equal')
plt.show()
