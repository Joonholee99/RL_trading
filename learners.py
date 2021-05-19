# 다양한 강화학습 방식을 수행하기 위한 학습기 클래스들을 가지는 모듈
import os
import keras
# print(os.environ.keys())
import logging
import abc
import collections
import threading
import time
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent 
from networks import Network, DNN
from visualizer import Visualizer 

class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method = 'rl', stock_code = None, chart_data = None,
                training_data = None, min_trading_unit = 1, max_trading_unit = 2,
                delayed_reward_threshold = 0.05, net = 'dnn', num_steps = 1, lr = 0.001,
                value_network = None, policy_network = None, output_path = '', reuse_models = True):
                
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit > min_trading_unit
        assert num_steps > 0
        assert lr > 0

        #강화학습 기법 설정
        self.rl_method = rl_method

        #환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)

        # 에이전트 설정
        self.agent = Agent(self.environment, min_trading_unit= min_trading_unit, max_trading_unit= max_trading_unit, 
                            delayed_reward_threshold = delayed_reward_threshold)
        
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1

        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM

        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]

        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models

        #가시화 모듈
        self.visualizer = Visualizer()

        # 메모리
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        
        # 에포크 관련 정보
        self.loss = 0
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

        #로그 등 출력 경로
        self.output_path = output_path

    def init_value_network(self, shared_network = None, activation = 'linear', loss='mse'):
        if self.net == 'dnn':
            self.value_network = DNN(input_dim = self.num_features, output_dim = self.agent.NUM_ACTIONS, lr=self.lr,
                shared_network = shared_network, activation = activation, loss=loss)

        
        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)

    def init_policy_network(self, shared_network=None, activation = 'sigmoid', loss = 'mse'):
        if self.net == 'dnn':
            self.policy_network = DNN(input_dim = self.num_features, output_dim = self.agent.NUM_ACTIONS, lr = self.lr, shared_network = shared_network, activation = activation, loss = loss)

        if self.reuse_models and os.path.exist(self.policy_network_path):
            self.policy_network.load_model(model_path = self.policy_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1

        #환경 초기화
        self.environment.reset()
        
        # init Agent
        self.agent.reset()
        
        # init visualizer
        self.visualizer.reset()
        
        # init memory
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        
        # init epch
        self.loss = 0
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx +1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass

    def update_networks(self, batch_size, delayed_reward, discount_factor):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch(batch_size, delayed_reward, discount_factor)
        
        if len(x) >0 :
            loss = 0
        
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x,y_value)
            
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x,y_policy)
            return loss
        return None

    def fit(self, delayed_reward, discount_factor):
        

    

    

