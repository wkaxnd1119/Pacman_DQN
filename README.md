# Reinforcement Learning by Pytorch

Using Reinforcement Learning(DQN) to play Pacman. 

![image](https://user-images.githubusercontent.com/85381860/143868194-a6a59d24-0195-4d4f-8e11-040de86aa3cd.png)



### ################################설명################################
환경 및 구현 알고리즘 설명
1. DQN Network 설정
DQN 학습 Network를 위해 Fully Connected Layer 를 3개 (Input, Hidden, Output) 로 간단하게 설정하였다. Input Layer의 size는 (11 x 32), Hidden Layer의 size는 (32 x 64), Output Layer의 size는 (64 x 4)다. 
Loss function은 pytorch의 L1loss를 구하는 함수 nn.SmoothL1Loss()를 사용하였다. 
Optimization 기법은 Adam을 사용하였고 Learning Rate은 주어진 값(0.0005)를 사용하였다. 
2. Input Variable
Input Variable으로 아래와 같이 총 5가지로 구성하였다.  
(1) 현재 팩맨의 위치: (x,y) 2개 input
(2) 팩맨가 가장 가까이 있는 음식의 위치: (x,y) 2개 input
(3) Ghost 의 위치: (x,y) 2개 input
(4) 팩맨 위치에서의 벽 위치: (a,b,c,d) 4개 input
(5) Ghost가 겁 먹었는지의 여부: True or False 1개 input
상기 5가지의 변수의 값을 나열하면 총 11가지의 Input Size가 나오게 된다. 
3. Output 
Output은 State에 따른 Argmax한 Action값을 택하기 때문에 0~3까지 총 4가지 경우의 수를 추출. 
4. Preprocessing 
Input Variable 5가지를 전처리 하기 위해 Pacman.py의 Gamestate 안의 함수들을 이용 
(1) 팩맨의 위치: state.getPacmanPosition() 함수로 호출 후 튜플 형태로 저장
(2) 고스트 위치: state.getGhostPositions() 함수로 호출 후 튜플 형태로 저장
(3) Food 위치: 팩맨의 위치와 Food의 위치를 L1 Distance 방식으로 거리를 구한 뒤 둘 중 Min 값을 호출 후 저장. 만약 Food가 하나만 남을 경우 남은 Food의 위치를 부르고 Food가 다 없어질 경우 게임이 끝난 경우이기에 팩맨의 위치를 호출
(4) 벽 위치: 팩맨 위치에 따른 벽의 위치를 확인하기 위해 state.hasWall() 함수를 이용한다.
팩맨의 상하좌우 위치 좌표를 넣으면 True or False 값으로 나오는데 True면 1, False 면 0을 Return 시켜 List 형태로 저장하여 값을 반환
(5) 고스트가 겁 먹었는지 여부: state.getScaredTimer()를 통해 True면 1, False 면 0을 반환
상기 항목들을 모두 전처리 한 후 각 int 형태의 숫자들을 한 List에 추가한다. 즉, size가 11개가 되는 list가 생성된다. 
5. Epsilon 업데이트
Epsilon의 초기 값은 기존에 주어진 0.8로 시작하며 0.015 x (에피소드 수 / 100) 값을 step마다 빼준다. 
6. Replay Memory 
매 step별로 action이 있을 경우에만 state, action, reward, next_state, done 을 self.replay_memory에 저장해 주었다. self.replay_memory는 size 50,000 인 deque 이며 size를 넘길 경우 popleft()를 통해 가장 먼저 저장된 값을 지우는 FIFO 방식으로 설정했다.
7. Sampling & Training
Smapling은 Training 단계에서 Replay Memory에 저장되어 있는 정보를 불러온다. Sampling Size는 32로 설정하였으며 Sampling 함수 안에서 각 값들을 List 형태로 불러온 뒤 Sampling Size만큼 합치고 그 합친 값들을 Torch.tensor로 변형시켜 DQN network에 학습시킨다. 
DQN 모델은 Predict와 Target 두 개로 나누어 Predict 모델만 학습시킨다. 다만 학습이 계속되면서 Step이 TARGET_UPDATE_ITER(400 step) 에 도달 시 아래 함수와 같이 Target 모델을 Predict 모델로 Update 시킨다. 
self.q_target.load_state_dict(self.pred_q.state_dict())

 






<DQN 성능>
 
