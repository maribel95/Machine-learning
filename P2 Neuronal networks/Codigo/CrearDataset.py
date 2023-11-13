import os
from torchvision import transforms as T
from torch.utils.data import  random_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.cuda
#   ----------------------------------        TRATAMIENTO DATOS     ------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
MABIMI = "Mabimi"
# Creamos listas vacías para almacenar los tensores de las imágenes y las máscaras
sources = []
masks = []

# Las imágenes están en carpetas
for folder in os.listdir(MABIMI):

    # Se valida que sea una carpeta.
    if os.path.isdir(os.path.join(MABIMI, folder)):

        # Se accede a las imágenes.
        for image in os.listdir(os.path.join(MABIMI, folder)):

            # Buscamos mask.jpg para meterla en su lista correspondiente
            if image.endswith('mask.jpg'):

                # Se obtiene la ruta completa de la imagen.
                image_path = os.path.join(MABIMI, folder, image)
                masks.append(image_path)
            # Lo mismo para las imagenes
            elif image.endswith('source.jpg'):

                # Se obtiene la ruta completa de la imagen.
                image_path = os.path.join(MABIMI, folder, image)
                sources.append(image_path)


#   ----------------------------------        DATASET     ------------------------------------
class Cell_Dataset(Dataset):
    def __init__(self, data, masks=None, img_transforms=None, mask_transforms=None):
        self.img_transforms = img_transforms # Transformación de las imagenes
        self.mask_transforms = mask_transforms # Transformación de las máscaras

        self.images = sorted(data) # Ordenamos las imagenes
        self.masks = sorted(masks) # Ordenamos las máscaras

    def __len__(self):
        return len(self.images) # Total imagenes del dataset

    def __getitem__(self, idx):
        # Aplicamos la transformación a cada imagen
        image_name = self.images[idx]
        img = Image.open(image_name)
        img = self.img_transforms(img)
        # Aplicamos la transformación a cada máscara
        mask_name = self.masks[idx]
        mask = Image.open(mask_name)
        mask = self.mask_transforms(mask)

        mask_max = mask.max().item() # Obtenemos el valor único más grande de las máscaras
        mask /= mask_max # Normalizamos dividiendo los valores de las máscaras por dicho valor

        return img, mask

#   ----------------------------------        TRANSFORMACIÓN     ------------------------------------
transform_data = T.Compose([
                T.Resize([224, 224]), # Redimensionamos las imagenes a un tamaño más asequible
                T.ToTensor() ])       # Las convertimos a tensor para poder trabajar con ellas

#   ----------------------------------          DATALOADER       ------------------------------------
# Creamos el dataset completo, con las rutas de las imagenes, de las máscaras y la transformación que aplicaremos
# a ambas
full_dataset = Cell_Dataset(sources,
                           masks,
                           img_transforms=transform_data,
                           mask_transforms=transform_data)


TRAIN_SIZE = int(len(full_dataset)*0.8) # Queremos que el 80% de los datos sean de entrenamiento
TEST_SIZE = len(full_dataset) - TRAIN_SIZE # El 20% será para el test
# Realizamos una división aleatoria de los datos de entrenamiento y los datos de test
train_dataset, test_dataset = random_split(full_dataset, [TRAIN_SIZE, TEST_SIZE])

print(len(train_dataset), len(test_dataset))


# Cargamos los datos en lotes
def crearDataLoaders(BATCH_SIZE):
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)