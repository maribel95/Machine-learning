import os

import cv2
import matplotlib.pyplot as plt

MABIMI = "Mabimi"
# Las imágenes están en carpetas
for folder in os.listdir(MABIMI):

    # Se valida que sea una carpeta.
    if os.path.isdir(os.path.join(MABIMI, folder)):

        # Se accede a las imágenes.
        for image in os.listdir(os.path.join(MABIMI, folder)):

            # Buscamos mask.jpg (vamos a normalizarla).
            if image.endswith('mask.jpg'):
                # Carga la imagen de la carpeta
                # Se obtiene la ruta completa de la imagen.
                image_path = os.path.join(MABIMI, folder, image)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # Aplica el thresholding para conseguir una imagen binaria
                ret, bin_img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

                # Guarda la imagen binaria
                cv2.imwrite(image_path, bin_img)

