import matplotlib.pyplot as plt

def mostrarCelulaEjemplo(model, valid_loader):
    # Ponemos el modelo en modo evaluación
    model.eval()
    train_iter = iter(valid_loader)
    images, labels = next(train_iter)
    output = model(images)

    output = output.squeeze().detach().numpy()
    output[output <= 0.5] = 0
    output[output > 0.5] = 1
    # Mostrar la máscara generada
    plt.imshow(output, cmap='gray')
    plt.show()

    # Seleccionar la primera imagen
    first_image = images[0]
    first_mask = labels[0]

    # Mostrar la primera máscara
    plt.imshow(first_mask[0], cmap='gray')
    plt.show()
    # Mostrar la primera imagen
    plt.imshow(first_image[0], cmap='gray')
    plt.show()
