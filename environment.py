class Environment:
    PRICE_INDEX = 4 # place of Last price

    def __init__(self, char_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1

    def reset(self):