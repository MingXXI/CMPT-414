import cv2

im = cv2.imread("test.tif");

cv2.imshow("test image", im);

im_median = cv2.medianBlur(im,3);

cv2.imshow("test median", im_median)

cv2.waitKey(10000);

cv2.destroyAllWindows()
