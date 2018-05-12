# importing packages for this tutorial
import cv2
import numpy
from matplotlib import pyplot as plt

# using grascale because need to go from color to grayscale to get the information accurately
img = cv2.imread('watch.jpg', cv2.IMREAD_GRAYSCALE)

# shows the image
cv2.imshow('image', img)
# waits for a key to be pressed and then destroys the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([]) # to hide tick values on x, y axis
# plt.plot([200, 300, 400], [100, 200, 300], 'c', linewidth=5)
# plt.show()

cv2.imwrite('watchgray.png', img)