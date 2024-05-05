class fasterRCNNloss():

    def __init__(self, scaling):
        self.scaling = scaling


    def __call__(self, model_output):
        loss_dict = model_output[1]
        losses = 0
        for key, loss in loss_dict.items():
            losses = losses + loss * int(self.scaling[str(key)])
        return losses