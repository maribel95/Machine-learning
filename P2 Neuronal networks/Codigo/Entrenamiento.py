
import torch
from torch.optim import Adam
from FuncionPerdida import DiceLoss
from tqdm import tqdm
import pylab as pl
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda


#   ---------------------------------------        ENTRENAMIENTO    ------------------------------------------



def entrenamiento(model, device, train_loader, valid_loader, ruta):
    # Número de iteraciones que entrenaremos el modelo
    num_epochs = 100
    # Nuestro modelo es la UNET
    #model = mmodel.to(device)
    # Definimos nuestro optimizador y la función de pérdida.
    # Adam: es un algoritmo de optimización estocástico basado en gradiente
    # El optimizador actualiza los pesos de los parámetros a partir de los resultados de la función de pérdida
    optim = Adam(model.parameters(), lr=1e-4)  # Podemos probar con SGD
    # DiceLoss: Mide la similitud entre dos distribuciones de probabilidad.
    # La función de pérdida mide el desempeño del modelo y obtiene un valor númerico que envía al optimizador
    criterion = DiceLoss()  # Podemos probar con Cross Entropy(aunque es para clasificación)
    # Vectores para poder comparar la pérdida del entrenamiento y de la validación.
    # De esta manera podremos hacer un análisis del rendimiento del modelo a través de las épocas.
    t_loss = np.zeros((num_epochs))
    v_loss = np.zeros((num_epochs))
    # Skin para mostrar los datos más bonitos(barra de carga fachera)
    pbar = tqdm(range(1, num_epochs + 1))  # tdqm permet tenir text dinàmic

    # ----- Comprobación de la red para ver si el tensor de salida es equivalente al de entrada
    def test():
        x = torch.randn((1, 3, 224, 224))
        return model(x)

    preds = test()
    print(preds.shape)

    for epoch in pbar:
        # Inicializamos la pérdida de entrenamiento a 0
        train_loss = 0
        # Inicializamos la pérdida de validación a 0
        val_loss = 0

        model.train()  # Ponemos el modelo en modo entrenamiento
        # Iteramos sobre los datos de entrenamiento, entrenaremos cada batch uno por uno
        for batch_num, (input_img, target) in enumerate(train_loader, 1):  # 1 porque empieza en el batch 1
            input_img = input_img.to(device)  # Pasamos source a CPU o GPU
            target = target.to(device)  # Lo mismo con la máscara

            output = model(input_img)  # Obtenemos resultado segmentación
            loss = criterion(output, target)  # Calculamos error del modelo
            loss.backward()  # Calcula el gradiente de la pérdida con respecto a los pesos del modelo
            optim.step()  # Actualiza los pesos del modelo basados en los gradientes calculados en el paso anterior
            optim.zero_grad()  # Establece el gradiente a cero para evitar la acumulación de gradientes en cada iteración

            train_loss += loss.item()

        model.eval()  # Ponemos el modelo en modo validación
        # Desactiva el seguimiento de gradientes durante la validación para ahorrar memoria
        with torch.no_grad():
            # Recorremos el conjunto de validación
            for input_img, target in valid_loader:
                input_img = input_img.to(device)
                target = target.to(device)

                output = model(input_img)
                loss = criterion(output, target)
                val_loss += loss.item()

        # RESULTADOS
        # Hace la media de la pérdida en el entrenamiento
        train_loss /= len(train_loader)
        # Guardamos en el vector de pérdidas el valor de pérdida la época actual
        t_loss[epoch - 1] = train_loss
        # Igual pero con la validación
        val_loss /= len(valid_loader)
        v_loss[epoch - 1] = val_loss

        # VISUALIZACIÓN DINÁMICA MODELO
        plt.figure(figsize=(10, 5))  # Tamaño de la figura 10x5
        # Mostrar valores de t_loss hasta el epoch-1 y etiquetamos como "entrenamiento"
        pl.plot(t_loss[:epoch - 1], label="Entrenamiento")
        # Mostrar valores de v_loss hasta el epoch-1 y etiquetamos como "Validación"
        pl.plot(v_loss[:epoch - 1], label="Validación")
        # Leyenda para identificar cada una de las líneas.
        pl.legend()
        # Limitar el eje x desde 0 hasta num_epochs.
        pl.xlim(0, num_epochs)

        # Mostrar gráfica en tiempo real en Jupyter NoteBook
        display.clear_output(wait=True)  # Borrar salida previa
        display.display(pl.gcf())  # Mostramos figura actual
        plt.close()  # Cierra figura
        # Print para saber
        pbar.set_description(f"Epoch:{epoch} Training Loss:{train_loss} Validation Loss:{val_loss}")
        # Guardamos los pesos del modelo
        torch.save(model.state_dict(), ruta)
