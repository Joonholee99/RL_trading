import numpy as np 
import utils

class Agent:
    STATE_DIM = 2 # Agent의 상태 -> 1. 주식 보유 비율 / 2. 포트폴리오 가치 비율
    TRADING_CHARGE = 0.00015 # 0.015% of charge
    TRADING_TAX = 0.0025 # 0.025% of charge

    # Action 
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2

    # From NN probability
    ACTIONS = [ACTION_BUY,ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)

    def __init__(self, environment, min_trading_unit = 1, max_trading_unit =2, delayed_reward_threshold = 0.05):

        # 환경 참조
        self.environment = environment

        self.min_trading_unit = min_trading_unit # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit # 최대 단일 거래 단위
        
        # 지연보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold

        # Agent 클래스 속성
        self.initial_balance = 0 # 초기 자본금
        self.balance = 0 # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식수 
        self.portfolio_value = 0 # PV = balance + num_stocks * (주식 가격)
        self.base_portfolio_value = 0 # 직전 학습 시점의 PV
        self.num_buy = 0 # 매수 횟수
        self.num_sell = 0 # 매도 횟수
        self.hold = 0 # 홀딩 횟수
        self.immediate_reward = 0 #즉시 보상
        self.profitloss = 0 # 현재 손익
        self.base_profitloss = 0 # 직전 지연 보상 이후 손익
        self.exploration_base = 0 # 탐험 행동 결정 기준

        # Agent 클래스의 상태
        self.ratio_hold = 0 #주식 보유 비율
        self.ratio_portfolio_value = 0 # 포트폴리오 가치 비율

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0 
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self):
        self.exploration_base = 0.5 + np.random.rand() / 2

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = (self.portfolio_value / self.base_portfolio_value)

        return(self.ratio_hold, self.ratio_portfolio_value)


    # 행동 결정 검사 함수
    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0

        pred = pred_policy
        if pred is None:
            pred = pred_value
        
        if pred is None:
            #예측 값이 없으면 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우, 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            if np.random.rand() < self.exploration_base: #exploration_base가 1에 가까울수록 더 많이 BUY
                action = self.ACTION_BUY
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1
        else:
            exploration = False
            action = np.argmax(pred)
            
        confidence = 0.5
        
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    # 행동 유효성 검사 함수
    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (1+self.TRADING_CHARGE) *self.min_trading_unit:

                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        
        return True


    def decide_trading_unit(self, confidence):
        # Confidence에 따라서 주식을 더 사거나 함 -> confidence 100이면 max, 0이면 min 개수 만큼 구매
        if np.isnan(confidence):
            return self.min_trading_unit
        
        added_trading = max(min(int(confidence * (self.max_trading_unit - self.min_trading_unit)), self.max_trading_unit - self.min_trading_unit),0)

        return self.min_trading_unit + added_trading

    def act(self, action, confidence):
        # 결정한 행동을 수행하는 module

        if not self.validate_action(action):
            action = Agent.ACTION_HOLD
        
        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화 -> 즉시 보상은 행동할 때마다 결정되기 때문에 초기화해야함
        self.immediate_reward = 0

        # BUY
        if action == Agent.ACTION_BUY:
            trading_unit = self.decide_trading_unit(confidence)
            balance = (self.balance - curr_price * (1 + self.TRADING_CHARGE)*(trading_unit))

            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대 매수
            if balance <0 :
                trading_unit = max(min(int(self.balance/(curr_price)*(1+self.TRADING_CHARGE)), self.max_trading_unit),self.min_trading_unit)
            
            # 수수료 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1+self.TRADING_CHARGE) * (trading_unit)
            if invest_amount > 0:
                self.balance -= invest_amount
                self.num_stocks += trading_unit
                self.num_buy += 1

        
        # SELL
        elif action == Agent.ACTION_SELL:
            #매도 단위 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 매도 단위가 보유 주식수보다 크지 않도록 제한
            trading_unit = min(trading_unit, self.num_stocks)

            #매도
            invest_amount = curr_price * (1-(self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit

            if invest_amount >0:
                self.num_stocks -= trading_unit
                self.balance += invest_amount
                self.num_sell += 1

        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1

        
        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = ((self.portfolio_value - self.intial_balance)/self.initial_balance)

        # 즉시 보상 -> 수익률
        self.immediate_reward = self.profitloss

        # 지연 보상 -> 익절, 손절 기준
        delayed_reward = 0

        self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value)/self.base_portfolio_value)
        
        # 기준 지연 보상 이상, 이하인 경우!
        if self.base_profitloss > self.delayed_reward_threshold or self.base_profitloss < -self.delayed_reward_threshold:
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        
        else:
            delayed_reward = 0

        return self.immediate_reward, delayed_reward