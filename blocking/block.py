
class Block:

    def __init__(self, value, raw):
        self.value = value
        self.raw_value = raw
        self.cnn_scores = {}
        self.fw_reinforcement_score = {}
        self.bw_reinforcement_score = {}
        self.clear()

    def set_anchor(self,attr,prob,freq):
        self.attr = attr
        self.anchor_prob = prob
        self.freq = freq
        self.is_anchor = True

    def clear(self):
        self.attr = 'none'
        self.anchor_prob = 0
        self.freq = 0
        self.is_anchor = False

    def get_top_fw_reinforcement_score(self):
        max = 0
        attr = 'none'
        for k,v in self.fw_reinforcement_score.items():
            if v > max:
                max = v
                attr = k
        return (max,attr)

    def get_top_bw_reinforcement_score(self):
        max = 0
        attr = 'none'
        for k,v in self.bw_reinforcement_score.items():
            if v > max:
                max = v
                attr = k
        return (max,attr)
