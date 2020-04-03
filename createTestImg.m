A=rand(5,7);
left=A(:,1:5);
right=A(:,2:6);
imwrite(mat2gray(left),'test1.png');
imwrite(mat2gray(right),'test2.png');
w=imread('test1.png');
