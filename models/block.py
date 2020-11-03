
class Block:

    def __init__(self, value, raw):
        self.value = value
        self.raw_value = raw
        self.clear()

    def set_anchor(self,label,prob,freq):
        self.label = label
        self.anchor_prob = prob
        self.freq = freq

    def clear(self):
        self.label = ''
        self.anchor_prob = 0
        self.freq = 0

    def is_anchor(self):
        return self.anchor_prob > 0
