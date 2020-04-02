import cv2
import numpy as np

# im = cv2.imread("test.tif");

# cv2.imshow("test image", im);

# im_median = cv2.medianBlur(im,3);

# cv2.imshow("test median", im_median)

# cv2.waitKey(10000);

# cv2.destroyAllWindows()


from PIL import Image

test = Image.open('image_left.png')
width,height = test.size
test = test.convert("L")
data = np.array(test)
print(type(data))
print(type(test))
new_data=np.reshape(data,(height,width))


w,h = test.size
print('w is', w, 'h is', h)
print(data.size)
w,h = new_data.shape
print('new data')
print('w is', w, 'h is', h)