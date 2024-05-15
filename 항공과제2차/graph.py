import matplotlib.pyplot as plt

# 비율 값과 해당 항목 이름
ratios = [0.01085, 0.17421, 0.00009, 0.03918, 0.00719, 0.79962, 0.01599, 0.00095]
labels = ['AAP', 'CGO', 'GAA', 'HAA', 'LAA', 'OFA', 'UAL', 'UOC']

# 그래프 생성
plt.bar(labels, ratios)

# 비율 값 표시
for i in range(len(labels)):
    plt.text(i, ratios[i], f'{ratios[i]*100:.2f}%', ha='center', va='bottom')


# 그래프 출력
plt.show()
