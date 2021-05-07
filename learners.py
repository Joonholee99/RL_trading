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
                traiding_data = None, min_trading_unit = 1, max_trading_unit = 2,
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

                

    