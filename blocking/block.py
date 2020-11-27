
class Block:

    def __init__(self, value, raw):
        self.value = value
        self.raw_value = raw
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
