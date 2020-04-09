import cv2

im = cv2.imread('truedisp.row3.col3.pgm')
cv2.imwrite('Ground Truth.png',im)