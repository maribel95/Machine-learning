
from MetricasEvaluacion import DiceScore, Iou


def evaluacion(model, device, valid_loader):
    dice_score_test = 0
    iou_test = 0
    # Recorremos el conjunto de validación
    for input_img, target in valid_loader:
        input_img = input_img.to(device)  # Pasamos source a CPU o GPU
        target = target.to(device)  # Lo mismo con la máscara
        # Hacer una predicción con el modelo
        outputs = model(input_img)
        # Calcular el DiceScore para cada tensor en el batch
        dice_calculator = DiceScore()
        iou_calculator = Iou()
        dice_batch = dice_calculator.calculateDice(outputs, target)
        iou_batch = iou_calculator.calculateIou(outputs, target)

        # Agregar los resultados de DiceScore para el batch actual a la lista de resultados
        dice_score_test += dice_batch.item()
        iou_test += iou_batch.item()

    # Imprimir el Dice promedio para toda la evaluación del test
    print(f"DiceScore entrenamiento promedio: {dice_score_test / len(valid_loader)}")
    print(f"Iou entrenamiento promedio: {iou_test / len(valid_loader)}")