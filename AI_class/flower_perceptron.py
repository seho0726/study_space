# 뉴런의 출력 계산 함수
def calculate(input):
    global weights
    global bias
    activation = bias   # 바이어스
    for i in range(4):  # 입력신호 총합 계산
        activation += weights[i] * input[i]

    if activation >= 0.0:   # 스텝 활성화 함수
        return 1.0
    else:
        return 0.0


#학습 알고리즘
def train_weights(x,y,l_rate,n_epoch):       # fit함수랑 같은것 임
    global weights
    global bias
    for epoch in range(n_epoch):            # 에포크 반복
        sum_error = 0.0
        for row, target in zip(x,y):        # 데이터셋을 반복
            actual = calculate(row)         # 실제 출력 계산
            error = target - actual         # 실제 출력 계산
            bias = bias + l_rate * error    # 에러를 줄이는 방향으로 계산
            sum_error += error**2           # 오류의 제곱 계산
            for i in range(4):              # 가중치 변경
                 weights[i] = weights[i] + l_rate * error * row[i] # 학습률이 기울기가 확 바뀌지 않도록 조정해주는 것 # 미분방정식을 풀어 쓴것을 의미
            print(weights, bias)            # 경사하강법
        print('에포크 번호 = %d, 학습률 = %.3f, 오류 = %.3f' % (epoch, l_rate, sum_error))
    return weights

def r_test(X_test, y_test):
    global weights
    global bias
    correct = 0
    wrong = 0
    for row, target in zip(X_test, y_test):
        actual = calculate(row)
        if target == actual:
            correct = correct+1
        else:
            wrong = wrong+1
    print("정답 : ", correct)
    print("오답 : ", wrong)

#AND 연산 학습 데이터셋, 샘플과 레이블이다.

x_training = [
     [4.4, 3, 1.3, 0.2],
     [5.1, 3.4, 1.5, 0.2],
     [5, 3.5, 1.3, 0.2],
     [4.5, 2.3, 1.3, 0.3],
     [4.4, 3.2, 1.3 ,0.2],
     [6.1, 2.8, 4.7, 1.2],
     [6.4, 2.9, 4.3, 1.3],
     [6.6, 3, 4.4, 1.4],
     [6.8, 2.8, 4.8, 1.4],
     [6.7, 3, 5, 1.7] ]

y_training=[0,0,0,0,0,
            1,1,1,1,1 ]

x_test = [
     [5.4, 3.7, 1.5, 0.2],
     [4.8, 3.4, 1.6, 0.2],
     [4.8, 3, 1.4, 0.1],
     [4.3, 3, 1.1, 0.1],
     [5.8, 4, 1.2,0.2],
     [5.7, 4.4, 1.5, 0.4],
     [5.4, 3.9, 1.3, 0.4],
     [5.2, 2.7, 3.9, 1.4],
     [5, 2, 3.5, 1],
     [5.9, 3, 4.2, 1.5],
     [6, 2.2, 4, 1],
     [6.1, 2.9, 4.7, 1.4],
     [5.6, 2.9, 3.6, 1.3],
     [6.7, 3.1, 4.4, 1.4] ]

y_test = [0,0,0,0,0,0,0,
            1,1,1,1,1,1,1 ]

#가중치와 바이어스 초기값
weights = [0.05, 0.05, 0.05, 0.05]
bias = 0.5  # 0이면 원점을 지나게하는것

l_rate = 0.1    #학습률
n_epoch = 100     #에포크 횟수
weights = train_weights(x_training,y_training,l_rate,n_epoch)
print(weights, bias)

r_test(x_test, y_test)

# 4차원의 경우에는 하이퍼플레인(초평면)을 알고있는게 중요하다.