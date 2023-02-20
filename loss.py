from torch import nn


class weighted_ner_loss:
    def __init__(self, weights):
        self.weights = weights
        self.criterion = nn.CrossEntropyLoss(weight = self.weights.float())

    def mlm_loss(self, prediction, target):
        prediction = prediction[0]
        loss = self.criterion(prediction.view(-1, 2), target.view(-1).long())
        return loss


class bce_loss:
    def __init__(self, weight_1):
        self.weights = weight_1
        self.criterion = nn.BCEWithLogitsLoss(pos_weight = weight_1)

    def mlm_loss(self, prediction, target):
        prediction = prediction[0]
        loss = self.criterion(prediction.view(-1), target.view(-1).long())
        return loss
