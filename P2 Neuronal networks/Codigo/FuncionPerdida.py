import torch.nn as nn
#   ---------------------------------------        FUNCIÓN DE PÉRDIDA     ------------------------------------------
# Nuestra función de pérdida utiliza la métrica Dice.
# Se calcula como la intersección entre los valores reales y predecidos dividido por la suma de los mismos.
class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 0.0 # Se utiliza para evitar divisiones por cero y se inicializa en 0.0.

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size() # Comprobamos que los tamaños coinciden
        # Salidas predichas del batch
        y_pred = y_pred[:, 0].contiguous().view(-1) # Aplanamos para tener un vector
        # Salidas reales del batch
        y_true = y_true[:, 0].contiguous().view(-1) # Aplanamos para tener un vector
        # Calculamos la intersección entre ellos
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc # Queremos el valor más próximo a 0(lo más pequeñito posible)