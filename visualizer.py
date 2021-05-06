import threading
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from mplfinance.original_flavor import candlestick_ohlc
from agent import Agent

lock = threading.Lock()

class Visualizer:
    COLORS = ['r','b','g']

    def __init__(self, vnet=False):
        self.canvas = None

        self.fig = None 

        self.axes = None 
        self.title = ''
    
    def prepare(self,chart_data,title):
        self.title = title
        with lock:
            self.fig, self.axes = plt.subplots(nrows=5, ncols = 1, facecolor = 'w', sharex=True)
            
            for ax in self.axes:
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)

                ax.yaxis.tick_right()

            self.axes[0].set_ylabel('Env.')
            x = np.arange(len(chart_data))

            # open, high, low, close 순서의 2차 배열
            ohlc = np.hstack((x.reshape(-1,1), np.array(chart_data)[:,1:-1]))

            # candlestick
            candlestick_ohlc(self.axes[0], ohlc, colorup='r',colordown = 'b')

            # 거래량 가시화
            ax = self.axes[0].twinx()
            volume = np.array(chart_data)[:,-1].tolist()
            ax.bar(x,volume,color='b',alpha = 0.3)

    def plot(self, epoch_str = None, num_epoches = None, epsilon = None, action_list = None,
            actions = None, num_stocks = None, outvals_value = [], outvals_policy = [], exps = None,
            learning_idxes = None, initial_balance = None, pvs = None):

            with lock:
                x = np.arange(len(actions))
                actions = np.array(actions)
                # 가치 신경망의 출력 배열
                outvals_value = np.array(outvals_value)
                # 정책 신경망의 출력 배열
                outvals_policy = np.array(outvals_policy)
                
                #초기 자본금 배열
                pvs_base = np.zeros(len(actions)) + initial_balance


                for action, color in zip(action_list, self.COLORS):
                    for i in x[actions == action]:
                        # 배경색으로 행동 표시
                        self.axes[1].axvline(i, color = color, alpha = 0.1)
                # 보유 주식 수 그리기
                self.axes[1].plot(x, num_stocks, '-k')

                # 차트 3 , 가치 신경망
                if len(outvals_value) > 0:
                    max_actions = np.argmax(outvals_value, axis=1)
                    for action, color in zip(action_list, self.COLORS):
                        # 배경
                        for idx in x:
                            if max_actions[idx] == action:
                                self.axes[2].axvline(idx, color=color, alpha=0.1)

                        # 가치 신경망 출력의 tanh 그리기
                        self.axes[2].plot(x, outvals_value[:,action], color=color, linestyle = '-')
                
                # 차트 4 , 정책 신경망
                # 탐험은 노란색
                for exp_idx in exps:
                    self.axes[3].axvline(exp_idx, color = 'y')
                # 행동을 배경으로 그리기
                _outvals = outvals_policy if len(outvals_policy) >0 else outvals_value

                for idx, outval in zip(x, _outvals):
                    color = 'white'

                    if np.isnan(outval.max()):
                        continue
                    if outval.argmax() == Agent.ACTION_BUY:
                        color = 'r' #매수는 붉은색
                    elif outval.armax() == Agent.ACTION_SELL:
                        color = 'b' #매도는 푸른색
                    self.axes[3].axvline(idx, color = color, alpha=0.3)

                if len(outvals_policy) >0:
                    for action, color in zip(action_list, self.COLORS):
                        self.axes[3].plot(x, outvals_policy[:,action],coor = color, linestyle = '-')

                # 차트 5, 포트폴리오 가치
                self.axes[4].axhline(
                    initial_balance, linestyle ='-', color = 'gray'
                )
                self.axes[4].fill_between(x, pvs, pvs_base, where=pvs > pvs_base, facecolor = 'r', alpha = 0.1)
                self.axes[4].fill_between(x, pvs, pvs_base, where=pvs < pvs_base, facecolor = 'b', alpha = 0.1)
                self.axes[4].plot(x,pvs,'-k')
                # 학습 위치 표시
                for learning_idx in learning_idxes:
                    self.axes[4].axvline(learning_idx, color = 'y')

                # 에포크 및 탐험 비율
                self.fig.suptitle('{} \n Epoch:{}/{} e = {:.2f}'.format(self.title, epoch_str, num_epoches, epsilon))
                # 캔버스 레이아웃 조정
                self.fig.tight_layout()
                self.fig.subplots_adjust(top=0.85)

    def clear(self, xlim):
        with lock:
            _axes = self.axes.tolist()
            for ax in _axes[1:]:
                ax.Cla()
                ax.relim()
                ax.autoscale()

            self.axes[1].set_ylabel("Agent") 
            self.axes[2].set_ylabel("V")
            self.axes[3].set_ylabel("P")
            self.axes[4].set_ylabel("PV")

            for ax in _axes:
                ax.set_xlim(xlim)
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)

                # X축 간격 일정
                ax.ticklabel_format(useOffset=False)
    def save(self, path):
        with lock:
            self.fig.savefig(path)               

                        
                