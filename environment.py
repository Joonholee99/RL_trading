class Environment:
    PRICE_INDEX = 4 # place of Last price

    def __init__(self, char_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        if len(self.chart_data) > self.idx:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]

            return self.observation
        
        return None
    
    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_INDEX]
        
        return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data

