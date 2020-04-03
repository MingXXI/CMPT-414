import cv2
import numpy as np

# im = cv2.imread("test.tif");

# cv2.imshow("test image", im);

# im_median = cv2.medianBlur(im,3);

# cv2.imshow("test median", im_median)

# cv2.waitKey(10000);

# cv2.destroyAllWindows()


from PIL import Image

test = np.uint8(np.zeros((3,10),dtype = int))
test += 20
print(test)
print(test.shape)