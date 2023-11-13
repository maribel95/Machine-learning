#   ---------------------------------------        IMPORTS     ------------------------------------------
import torch
import torch.nn as nn
from collections import OrderedDict

#   ---------------------------------------        UNET     ------------------------------------------

#Credits: https://github.com/mateuszbuda/brain-segmentation-pytorch
class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        # Las características de inicio
        features = init_features

        ## CODER
        """ La parte de codificación es responsable de extraer características de los datos de entrada y
         transformarlos en una representación de características más compacta y significativa.
        """
        # Aplicamos las dos capas de convolución
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        # Aplicamos un MaxPooling para reducir la dimensionalidad y que la red sea más invariante a la escala
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Aplicamos otras dos capas de convolución
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        # Y otro MaxPooling
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Aplicamos otras dos capas de convolución
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        # Maxpooling
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Aplicamos otras dos capas de convolución
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        # Maxpooling
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Parte de "abajo del todo" del modelo
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        ## DECODER
        """
            El decodificador toma la representación compacta y 
            la amplía de nuevo hasta la forma original de la entrada.
        """
        # Aplicamos convolución transpuesta: es la operación inversa al MaxPooling.
        # Se utiliza para aumentar la resolución de una imagen en lugar de disminuirla.
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        # Doble capa de convolución
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        # Operacion UpSampling
        dec4 = self.upconv4(bottleneck)
        # Concatenamos dec4 con la skiped connection enc4
        dec4 = torch.cat((dec4, enc4), dim=1) # En el eje 1
        # Aplicamos decoder
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1)) # Para retornar las probabilidades de que cada pixel pertenezca a una clase u otra

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict( # Diccionario que recuerda el orden de inserción
                [
                    # Aplicamos la primera convolución, con su normalización pertinente y capa de activación ReLU
                    (name + "conv1", nn.Conv2d(in_channels=in_channels,out_channels=features,kernel_size=3,padding=1,bias=False)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    # Aplicamos la segunda convolución, con su normalización pertinente y capa de activación ReLU
                    (name + "conv2",nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True))
                ]
            )
        )



#   ---------------------------------------        POTINET     ------------------------------------------


class PotiNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(PotiNet, self).__init__()
        # Las características de inicio
        features = init_features

        ## CODER
        """ La parte de codificación es responsable de extraer características de los datos de entrada y
         transformarlos en una representación de características más compacta y significativa.
        """
        # Aplicamos las dos capas de convolución
        self.encoder1 = PotiNet._block(in_channels, features, name="enc1")
        # Aplicamos un MaxPooling para reducir la dimensionalidad y que la red sea más invariante a la escala
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Aplicamos otras dos capas de convolución
        self.encoder2 = PotiNet._block(features, features * 2, name="enc2")
        # Y otro MaxPooling
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Aplicamos otras dos capas de convolución
        self.encoder3 = PotiNet._block(features * 2, features * 4, name="enc3")
        # Maxpooling
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Aplicamos otras dos capas de convolución
        self.encoder4 = PotiNet._block(features * 4, features * 8, name="enc4")
        # Maxpooling
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Parte de "abajo del todo" del modelo
        self.bottleneck = PotiNet._block(features * 8, features * 16, name="bottleneck")

        ## DECODER
        """
            El decodificador toma la representación compacta y 
            la amplía de nuevo hasta la forma original de la entrada.
        """
        # Aplicamos convolución transpuesta: es la operación inversa al MaxPooling.
        # Se utiliza para aumentar la resolución de una imagen en lugar de disminuirla.

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        # Doble capa de convolución
        self.decoder4 = PotiNet._block((features * 8), features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = PotiNet._block((features * 4), features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = PotiNet._block((features * 2), features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = PotiNet._block(features, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        # Operacion UpSampling
        dec4 = self.upconv4(bottleneck)
        # Aplicamos decoder
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))  # Para retornar las probabilidades de que cada pixel pertenezca a una clase u otra

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(  # Diccionario que recuerda el orden de inserción
                [
                    # Aplicamos la primera convolución, con su normalización pertinente y capa de activación ReLU
                    (name + "conv1", nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    # Aplicamos la segunda convolución, con su normalización pertinente y capa de activación ReLU
                    (name + "conv2",nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True))
                ]
            )
        )


class XopiNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(XopiNet, self).__init__()
        # Las características de inicio
        features = init_features

        ## CODER
        """ La parte de codificación es responsable de extraer características de los datos de entrada y
         transformarlos en una representación de características más compacta y significativa.
        """
        # Aplicamos las dos capas de convolución
        self.encoder1 = XopiNet._block(in_channels, features, name="enc1")
        # Aplicamos un MaxPooling para reducir la dimensionalidad y que la red sea más invariante a la escala
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Aplicamos otras dos capas de convolución
        self.encoder2 = XopiNet._block(features, features * 2, name="enc2")
        # Y otro MaxPooling
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Aplicamos otras dos capas de convolución
        self.encoder3 = XopiNet._block(features * 2, features * 4, name="enc3")
        # Maxpooling
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Aplicamos otras dos capas de convolución
        self.encoder4 = XopiNet._block(features * 4, features * 8, name="enc4")
        # Maxpooling
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder5 = XopiNet._block(features * 8, features * 16, name="enc5")
        # Maxpooling
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Parte de "abajo del todo" del modelo
        self.bottleneck = XopiNet._block(features * 16, features * 32, name="bottleneck")

        ## DECODER
        """
            El decodificador toma la representación compacta y 
            la amplía de nuevo hasta la forma original de la entrada.
        """
        # Aplicamos convolución transpuesta: es la operación inversa al MaxPooling.
        # Se utiliza para aumentar la resolución de una imagen en lugar de disminuirla.
        self.upconv5 = nn.ConvTranspose2d(features * 32, features * 16, kernel_size=2, stride=2)
        # Doble capa de convolución
        self.decoder5 = XopiNet._block((features * 16) * 2, features * 16, name="dec5")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        # Doble capa de convolución
        self.decoder4 = XopiNet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = XopiNet._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = XopiNet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = XopiNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        bottleneck = self.bottleneck(self.pool5(enc5))
        # Operacion UpSampling
        dec5 = self.upconv5(bottleneck)
        # Concatenamos dec4 con la skiped connection enc4
        dec5 = torch.cat((dec5, enc5), dim=1) # En el eje 1
        # Aplicamos decoder
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1)) # Para retornar las probabilidades de que cada pixel pertenezca a una clase u otra

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict( # Diccionario que recuerda el orden de inserción
                [
                    # Aplicamos la primera convolución, con su normalización pertinente y capa de activación ReLU
                    (name + "conv1", nn.Conv2d(in_channels=in_channels,out_channels=features,kernel_size=3,padding=1,bias=False)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    # Aplicamos la segunda convolución, con su normalización pertinente y capa de activación ReLU
                    (name + "conv2",nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True))
                ]
            )
        )


