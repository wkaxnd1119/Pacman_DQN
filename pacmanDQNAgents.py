# import pacman game  

import sys

from pacman import Directions 
from pacmanUtils import * 
from game import Agent 
import game 

# import torch library 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 

  
#import other libraries 
import random 
import numpy as np 
import time 
from collections import deque 

  
# model parameters 

DISCOUNT_RATE = 0.99        # discount factor 
LEARNING_RATE = 0.0005      # learning rate parameter 
REPLAY_MEMORY = 50000       # Replay buffer 의 최대 크기 
LEARNING_STARTS = 5000 	    # 5000 스텝 이후 training 시작 
TARGET_UPDATE_ITER = 400   # update target network 
EPSILON_START = 0.8 
EPSILON_END = 0.01 

############################################### 

# Additional Template Codes                   # 

# You may or may not use below skeleton code. # 

############################################### 


class DQN(torch.nn.Module): 
    def __init__(self): 
        super(DQN, self).__init__() 
        self.fc1 = nn.Linear(4, 128) 
        self.fc2 = nn.Linear(128, 256) 
        self.fc3 = nn.Linear(256, 128) 
        self.fc4 = nn.Linear(128, 4) 
        
    def forward(self, x):
        # print('x: {}'.format(x))
        # print('shape of x: {}'.format(np.shape(x)))
        # print('type of x: {}'.format(type(x)))
        # print('fc1(x): {}'.format(self.fc1(x)))
        # print('Relu of fc1(x): {}'.format(F.relu(self.fc1(x))))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x))        
        x = self.fc4(x) 
        return x 
    

class ReplayMemory: 
    def __init__(self): 
        self.replay_memory = deque(maxlen=REPLAY_MEMORY) 

    def write(self, state, action, reward, next_state, done): 
        if len(self.replay_memory) >= REPLAY_MEMORY: 
            self.replay_memory.popleft() 
        self.replay_memory.append((state, action, reward, next_state, done)) 
        return  


    def sample(self, batch_size): 
         
        minibatch = random.sample(self.replay_memory, batch_size) 
        batch_s, batch_a, batch_r, batch_n, batch_t = [], [], [], [], []  

        trigger = True

        for transition in minibatch:
            s,a,r,n,t = transition 
            if trigger == True:
                batch_s = s
                batch_a.append([a]) 
                batch_r.append([r]) 
                batch_n = n
                batch_t.append([t]) 
                
                trigger = False
                
            else:
                
                batch_s = torch.cat([batch_s, s], dim = 0)
                batch_a.append([a]) 
                batch_r.append([r]) 
                batch_n = torch.cat([batch_n, n], dim = 0)
                batch_t.append([t]) 
            

            
        action = torch.tensor(batch_a, dtype=torch.int64)
        reward = torch.tensor(batch_r, dtype=torch.float)
        t = torch.tensor(batch_t, dtype=torch.float)
        
        # print('shape of action: {}'.format(np.shape(action)))
        # print('shape of reward: {}'.format(np.shape(reward)))
        # print('type t: {}'.format(type(t)))
        return batch_s, action, reward, batch_n, t
    
    # torch.tensor(batch_s, dtype=torch.float), torch.tensor(batch_a, dtype=torch.int64),\
    #         torch.tensor(batch_r, dtype=torch.float), torch.tensor(batch_n,dtype=torch.float), \
    #             torch.tensor(batch_t, dtype=torch.float) 

               
    def size(self): 
        return len(self.replay_memory) 
          
  
############################################### 
# End of Additional Template Codes            # 
############################################### 

  

class PacmanDQN(PacmanUtils): 

    def __init__(self, args):         
        print("Started Pacman DQN algorithm") 
        #print(args) 
        self.double = args['double'] 
        self.multistep = args['multistep'] 
        self.n_steps = args['n_steps'] 
        self.model = args['model'] 
        self.k = 1 
        # self.pred_q = DQN()
        # self.q_target = DQN()
        
        self.count_steps = 0
        self.trained_model = args['trained_model'] 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.trained_model:     
            mode = "Test trained model" 
        else: 
            mode = "Training model" 
        
       
        print("=" * 100) 
        print("Double : {}    Multistep : {}/{}steps    Train : {}    Test : {}    Mode : {}     Model : {}".format( 
                self.double, self.multistep, self.n_steps, args['numTraining'], args['numTesting'], mode, args['model'])) 
        print("=" * 100) 

        # Target 네트워크와 Local 네트워크, epsilon 값을 설정 
        if self.trained_model:  # Test 
            self.pred_q = torch.load(self.model).to(self.device)
            self.epsilon = 0 
        else:                   # Train 
            self.epsilon = EPSILON_START  # epsilon init value 
            self.pred_q = DQN().to(self.device)
            self.q_target = DQN().to(self.device)
            
            
        self.optimizer = optim.Adam(self.pred_q.parameters(), lr=LEARNING_RATE)
   


        # statistics 

        self.win_counter = 0       # number of victory episodes 
        self.steps_taken = 0       # steps taken across episodes 
        self.steps_per_epi = 0     # steps taken in one episodes    
        self.episode_number = 0 
        self.episode_rewards =[]        
        self.epsilon = EPSILON_START  # epsilon init value 
        self.r = ReplayMemory() 
        self.last_reward = 0


    def predict(self, state):  
        # state를 넣어 policy에 따라 action을 반환 (epsilon greedy) 
        # Hint: network에 state를 input으로 넣기 전에 preprocessing 해야합니다. 
        state = self.preprocess(state)
        # print(state)
        # state = torch.FloatTensor(state) 
        out = self.pred_q(state) 
       
        if np.random.rand(1) < self.epsilon: 
            act = np.random.randint(0, 4) 
        else: act = out.argmax().item() 
        self.action = act # save action 
        # print(act)
        return act 


    def update_epsilon(self): 
        # Exploration 시 사용할 epsilon 값을 업데이트 
        if not self.trained_model:
            self.epsilon = max(EPSILON_END , 0.8 - 0.014* (self.episode_number/(100))) #Epslion 크기를 최소 0.01 보장 



    def train(self): 
        # replay_memory로부터 mini batch를 받아 policy를 업데이트 
        if (self.steps_taken > LEARNING_STARTS): 
             s,a,r,n,t = self.r.sample(30)
             
             # get Q(s,a)
             state_action_values = self.pred_q(s).gather(1, a)
             
             # get V(s')
             next_state_values = self.q_target(n)
             
             #computing expected Q values
             next_state_values = next_state_values.detach().max(1)[0]
             next_state_values = next_state_values.unsqueeze(1)
             
             expected_state_action_values = (next_state_values * DISCOUNT_RATE) + r
             
             
             
             # q_out = self.pred_q(s)
             # # print('a: {}'.format(a))
             # # print('q_out: {}'.format(q_out))
             # q_a = q_out.gather(1,a) # get Q(s,a) 
             # max_q_prime = self.q_target(n).max(1)[0].unsqueeze(1)
             # # print('max prime:', max_q_prime)
             # target = r + DISCOUNT_RATE * max_q_prime * t
             
             			# calculate loss
             loss_function = torch.nn.SmoothL1Loss()
             self.loss = loss_function(state_action_values, expected_state_action_values)
             
 			# optimize model - update weights
             self.optimizer.zero_grad()
             self.loss.backward()
             self.optimizer.step()   
    
             
             # loss = F.smooth_l1_loss(q_a, target)
         
             # # calculate loss
             # self.optimizer.zero_grad()
             # loss.backward() #파라미터들의 Gradient 값 구하기 
             # self.optimizer.step() # update
             # print(loss)
      
   

    def reset(self): 
        # 새로운 episode 시작시 불리는 함수. 
        self.last_score = 0 
        self.current_score = 0 
        self.episode_reward = 0 
        self.episode_number += 1 
        self.steps_per_epi = 0 
    

    def final(self, state): 
        # episode 종료시 불리는 함수. 
        done = True 
        reward = self.getScore(state) 
        if reward >= 0: # not eaten by ghost when the game ends 
            self.win_counter +=1 
 
        

        self.step(state, reward, done) 
        self.episode_rewards.append(self.episode_reward) 
        win_rate = float(self.win_counter) / 500.0 
        avg_reward = np.mean(np.array(self.episode_rewards)) 
        
		# print episode information 
        if(self.episode_number%500 == 0): 
            print("Episode no = {:>5}; Win rate {:>5}/500 ({:.2f}); average reward = {:.2f}; epsilon = {:.2f}".format(self.episode_number, 
                                                                    self.win_counter, win_rate, avg_reward, self.epsilon)) 
            self.win_counter = 0 
            self.episode_rewards= [] 
            if(self.trained_model==False and self.episode_number%1000 == 0): 
                # Save model here 
                
                torch.save(self.pred_q, self.model) 
                pass 

    def preprocess(self, state): 

        def pacman_position(state):
            # print('shape of state: {}'.format(np.shape(state)))
            result = state.getPacmanPosition() 
            # print('state: {}'.format(state))
            
            return result 
        
        def ghost_position(state):
            result = state.getGhostPositions()
            return result
        
        def capsule_position(state):
            result = state.getCapsules()
            return result
        
        def food_position(state):
            result = state.getFood()
            position = []
            for i in range(7):
                for j in range(len(result[i])):
                    if result[i][j] == True:
                        position.append((i,j))         
            return position
            
        obs =[]
        obs.append([pacman_position(state)])
        obs.append(ghost_position(state))
        
        
        temp = []
        
        for i in range(len(obs)):
            for j in range(len(obs[i][0])):
                temp.append(obs[i][0][j])
        # print('shape of obs: {}'.format(np.shape(obs)))
        # print('temp: {}'.format(temp))
        temp = np.expand_dims(np.array(temp), 0)
        new_obs = torch.tensor(temp, dtype=torch.float)
  
        return new_obs


    def step(self, next_state, reward, done): 
        # next_state = self.state에 self.action 을 적용하여 나온 state 
        # reward = self.state에 self.action을 적용하여 얻어낸 점수.    
        if self.action is None: 
            self.state = self.preprocess(next_state)

        else: 
            self.next_state = self.preprocess(next_state)
            self.r.write(self.state, self.action, reward, self.next_state, done) 
            # print('\n\ncheck here1: {}'.format(next_state))
            self.state = self.next_state
            
        if reward > 20:
             self.last_reward = 10
        elif reward > 20:
             self.last_reward = 50
        elif reward < -10:
             self.last_reward = -500
        elif reward <0:
             self.last_reward = -1
        
        self.episode_reward +=  self.last_reward
        
   
        
        # next 
        self.episode_reward += reward 
        self.steps_taken += 1 
        self.steps_per_epi += 1 
        self.update_epsilon() 


        if(self.trained_model == False):
            self.train() 

            if(self.steps_taken % TARGET_UPDATE_ITER == 0): 
                self.q_target.load_state_dict(self.pred_q.state_dict())
                # print("n_step : {}".format(self.steps_taken))
                # UPDATE target network  
                pass 
        