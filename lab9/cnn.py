import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from skimage import color
from scipy import ndimage

# tworzymy tablice o wymiarach 128x128x3 (3 kanaly to RGB)
# uzupelnionÄ zerami = kolor czarny
data = np.zeros((128, 128, 3), dtype=np.uint8)


# chcemy zeby obrazek byl czarnobialy,
# wiec wszystkie trzy kanaly rgb uzupelniamy tymi samymi liczbami
# napiszmy do tego funkcje
def draw(img, x, y, color):
    img[x, y] = [color, color, color]


# zamalowanie 4 pikseli w lewym gĂłrnym rogu
draw(data, 5, 5, 100)
draw(data, 6, 6, 100)
draw(data, 5, 6, 255)
draw(data, 6, 5, 255)


# rysowanie kilku figur na obrazku
for i in range(128):
    for j in range(128):
        if (i-64)**2 + (j-64)**2 < 900:
            draw(data, i, j, 200)
        elif i > 100 and j > 100:
            draw(data, i, j, 255)
        elif (i-15)**2 + (j-110)**2 < 25:
            draw(data, i, j, 150)
        elif (i-15)**2 + (j-110)**2 == 25 or (i-15)**2 + (j-110)**2 == 26:
            draw(data, i, j, 255)

def apply_kernel_to_image(img, kernel, stride = 1):
    feature = convolve2d(img, kernel, boundary='symm', mode='same')[::stride, ::stride]
    return feature



kernel_vertical_borders = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
])

kernel_horizontal_borders = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])


def apply_linear_rectyfier(data):
    for x in data:
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = 0
    return data      


def apply_different_function(data):
    for x in data:
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = 0
            if x[i] > 255:
                x[i] = 255
    return data



vertical = apply_kernel_to_image(color.rgb2gray(data), kernel_vertical_borders, 1)
horizontal = apply_kernel_to_image(color.rgb2gray(data), kernel_horizontal_borders, 1)

# plt.imshow(apply_different_function(arr), cmap='gray')
# plt.imshow(apply_linear_rectyfier(arr), cmap='gray')
# plt.show()


fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(431)
ax1.imshow(data, 'gray')
ax1.set_title('Input image', fontsize=10)

ax2 = fig.add_subplot(432)
ax2.imshow(vertical, 'gray')
ax2.set_title("vertical borders recognition", fontsize=10)

ax3 = fig.add_subplot(433)
ax3.imshow(horizontal, 'gray')
ax3.set_title("horizontal borders recognition", fontsize=10)

ax1_2 = fig.add_subplot(434)
ax1_2.imshow(data, 'gray')
ax1_2.set_title('Input image', fontsize=10)


ax4 = fig.add_subplot(435)
ax4.imshow(apply_linear_rectyfier(horizontal), 'gray')
ax4.set_title("horizontal borders recognition with linear rectyfier", fontsize=10)

ax5 = fig.add_subplot(436)
ax5.imshow(apply_different_function(horizontal), 'gray')
ax5.set_title("horizontal borders recognition with different activating function", fontsize=10)

ax1_3 = fig.add_subplot(437)
ax1_3.imshow(data, 'gray')
ax1_3.set_title('Input image', fontsize=10)


ax6 = fig.add_subplot(438)
ax6.imshow(apply_linear_rectyfier(vertical), 'gray')
ax6.set_title("vertical borders recognition with linear rectyfier", fontsize=10)

ax6 = fig.add_subplot(439)
ax6.imshow(apply_different_function(vertical), 'gray')
ax6.set_title("vertical borders recognition with different activating function", fontsize=10)


ax1_4 = fig.add_subplot(4,3,10)
ax1_4.imshow(data, 'gray')
ax1_4.set_title('Input image', fontsize=10)

ax7 = fig.add_subplot(4,3,11)
ax7.imshow(ndimage.sobel(data), 'gray')
ax7.set_title('Image does not have any oblique lines\nsobel outputs black image', fontsize=10)


