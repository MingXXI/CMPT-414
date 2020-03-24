import cv2

im = cv2.imread("test.tif");

cv2.namedWindow("Test Window");

cv2.imshow("test image", im);

cv2.waitKey(100);

cv2.destroyAllWindows()