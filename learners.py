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
        # 배치 학습 데이터 생성 및 신경망 갱신
        if self.batch_size > 0 :
            _loss = self.update_networks(self.batch_size, delayed_reward, discount_factor)
            if _loss is not None:
                self.loss += abs(_loss)
                self.learning_cnt += 1
                self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0

    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] * (self.num_steps-1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps-1) + self.memory_num_stocks
        
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] * len(Agent.ACTIONS))] * (self.num_steps-1) + self.memory_value
        
        if self.policy_network is not None:
            self.memory_policy = [np.array([np.nan] * len(Agent.ACTIONS))] * (self.num_steps-1) + self.memory_policy
        
        self.memory_pv = [self.agent.initial_balance] * (self.num_steps - 1) + self.memory_pv

        self.visualizer.plot(epoch_str = epoch_str, num_epoches = num_epoches, epsilon = epsilon, action_list = Agent.ACTIONS,
        num_stocks = self.memory_num_stocks, outvals_value = self.memory_value, outvals_policy = self.memory_policy, exps = self.memory_exp_idx,
        learning_idxes = self.memory_learning_idx, initial_balance = self.agent.initial_balance, pvs = self.memory_pv)

        self.visualizer.save(os.path.join(self.epoch_summary_dir, 'epoch_summary_{}.png'.format(epoch_str)))
    
    #강화학습 실행 함수
    def run(self, num_epochs = 100, balance = 1000000, discount_factor = 0.9, start_epsilon = 0.5, learning=True):
        info = "[{code}] RL:{rl} Net:{net} LR:{lr}"\
            "DF : {discount_factor} TU: [{min_trading_unit}, "\
                "{max_trading_unit}] DRT:{delayed_reward_threshold}".format(code=self.stock_code, rl = self.rl_method, net=self.net,
                lr = self.lr, discount_factor = discount_factor, min_trading_unit = self.agent.min_trading_unit, max_trading_unit = self.agent.max_trading_unit,
                delayed_reward_threshold = self.agent.delayed_reward_threshold)
        #learning = True -> Save the model //// learning = False -> only simulation
        with self.lock:
            logging.info(info)
        
        # 시작시간
        time_start = time.time()

        # 가시화 준비
        self.visualizer.prepare(self.encironment.char_data, info)

        #가시화 저장할 폴더
        self.epoch_summary_dir = os.path.join(self.output_path, 'epoch_summary_{}'.format(self.stock_code))
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))
        
        # Agent 초기 자본금
        self.agent.set_balance(balance)

        # 학습 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        for epoch in range(num_epochs):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen = self.num_steps)

            # 환경, 에이전트, 신경망, 가시화, 메모리 최소화
            self.reset()

            # 학습을 진행할수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon * (1. - float(epoch) / (num_epochs-1))
                self.agent.reset_exploration()
            else:
                epsilon = start_epsilon
            
            while True:
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # num_steps만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))

                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))
                
                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = self.agent.decide_action(pred_value, pred_policy, epsilon)
                
                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                
        





