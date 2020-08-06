class MLDataModel():
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        self.features = [] # [(input_feature,label),...]

    # def encode(self,input_a, input_b)

    def add(self, input_a, input_b = None, label=None):
        pass
        # if(input_b is not None):
        #     input_feature = self.toBertIds(input_a,input_b)
        # else:
        #     input_feature = self.toBertIds(input_a)
        # self.features.append((input_feature,label))