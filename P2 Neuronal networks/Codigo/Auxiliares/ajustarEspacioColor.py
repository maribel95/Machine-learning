import cv2

# Carga la imagen de la carpeta
img = cv2.imread("Mabimi/cell_14/mask.jpg")

# Convierte la imagen de RGB a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Guarda la imagen en escala de grises
cv2.imwrite("Mabimi/cell_14/mask.jpg", gray_img)