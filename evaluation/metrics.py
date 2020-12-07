class Metrics:

    def __init__(self):
        self.precision = 0
        self.recall = 0
        self.f_measure = 0

    def calculate_f_measure(self):
        numerator = 2*self.precision*self.recall
        denominator = self.precision+self.recall
        self.f_measure = numerator / denominator
