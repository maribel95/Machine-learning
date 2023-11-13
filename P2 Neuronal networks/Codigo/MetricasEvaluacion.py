import torch.nn as nn

class DiceScore(nn.Module):

    def __init__(self):
        super(DiceScore, self).__init__()
        self.smooth = 0.0 # No tiene que ser un valor diferente a 0?

    def calculateDice(self, y_pred, y_true):

        assert y_pred.size() == y_true.size()

        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        result = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return result

class Iou(nn.Module):

    def __init__(self):
        super(Iou, self).__init__()
        self.smooth = 0.0 # No tiene que ser un valor diferente a 0?

    def calculateIou(self, y_pred, y_true):

        assert y_pred.size() == y_true.size()

        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum() - intersection
        result = (intersection + self.smooth) / (union + self.smooth)
        return result


