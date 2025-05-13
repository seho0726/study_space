# 뉴런의 출력 계산 함수
def calculate(input):
    global weights
    global bias
    activation = bias   # 바이어스
    for i in range(2):  # 입력신호 총합 계산
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
            error = target - actual         # 실제 출력 계산 # Bias가 미분 하는 방식도 적어내자. w = w- n*dE/dw (에타)
            bias = bias + l_rate * error    # 에러를 줄이는 방향으로 계산
            sum_error += error**2           # 오류의 제곱 계산
            for i in range(2):              # 가중치 변경
                weights[i] = weights[i] + l_rate * error * row[i] # 학습률이 기울기가 확 바뀌지 않도록 조정해주는 것 # 미분방정식을 풀어 쓴것을 의미
            print(weights, bias)
        print('에포크 번호 = %d, 학습률 = %.3f, 오류 = %.3f' % (epoch, l_rate, sum_error))
    return weights

def test(X_test, y_test):
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
'''
x_training = [ [1,3],
     [2,4],
     [3,3],
     [4,4],
     [5,2],
      [12,13],
      [20,40],
      [25,23],
      [45,18],
      [50,20] ]
'''
x_training = [ [-12,-4],
     [-4,-4],
     [-10,-8],
     [-18,-13],
     [-3,-12],
      [2,3],
      [4,10],
      [7,8],
      [17,9],
      [19,4] ]

y_training=[1,1,1,1,1,0,0,0,0,0]

# test data
'''
x_test=[ [3,5],
         [4,2],
         [7,9],
         [11,11],
         [27,11],
         [40,45] ]
'''
x_test=[ [-10,-1],
         [-1,-1],
         [-7,-12],
         [1,1],
         [10,1],
         [14,13] ]

y_test= [1,1,1, 0,0,0]

#가중치와 바이어스 초기값
weights = [0.05, 0.05]
bias = 0.9

l_rate = 0.1    #학습률
n_epoch = 100     #에포크 횟수
weights = train_weights(x_training,y_training,l_rate,n_epoch)
print(weights, bias)

test(x_test, y_test)
