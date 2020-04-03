# width(y) and height(x) is opposite
from matplotlib import pyplot
import numpy as np
from PIL import Image
# import PIL.image 	#require PIL package installation. 
import cv2 	#
# from energy import energy.py
import collections # check 2 list has same elements elements 
from itertools import permutations # for choosing alpha/beta randomly
import copy # to do deep copy
import time
def imageProcess(file):
	im=cv2.imread(file)
	height,width,color = im.shape
	#WIDTH is y-axis, HEIGHT is x-axis
	pixelNum=width*height
	im2=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	# data=im.getdata()
	data=np.array(im2,dtype='int')
	# new_data=np.reshape(data,(height,width))
	# use directory store neighbors of each vertex
	imEdge = {}
	imEdge[(0,0)]=[[1,0],[0,1]]

	##new version
	imEdge[(height-1,width-1)]=[[height-1,width-2],[height-2,width-1]]
	imEdge[(height-1,0)]=[[height-1,1],[height-2,0]]
	imEdge[(0,width-1)]=[[1,width-1],[0,width-2]]
	###
	for i in range(height-2):
		imEdge[(i+1,0)]=[[i,0],[i+2,0],[i+1,1]]
		imEdge[(i+1,width-1)]=[[i,width-1],[i+2,width-1],[i+1,width-2]]
	for j in range(width-2):
		imEdge[(0,j+1)]=[[0,j],[0,j+2],[1,j+1]]
		imEdge[(height-1,j+1)]=[[height-1,j],[height-1,j+2],[height-2,j+1]]
	for i in range(height-2):
		for j in range(width-2):
			imEdge[(i+1,j+1)]=[[i+1,j],[i,j+1],[i+1,j+2],[i+2,j+1]]
	return data,imEdge



	#origin
	# imEdge[(width-1,height-1)]=[[width-2,height-1],[width-1,height-2]]
	# imEdge[(0,height-1)]=[[1,height-1],[0,height-2]]
	# imEdge[(width-1,0)]=[[width-1,1],[width-1-2,0]]
	# ###
	# for i in range(height-2):
	# 	imEdge[(0,i+1)]=[[0,i],[0,i+2],[1,i+1]]
	# 	imEdge[(width-1,i+1)]=[[width-1,i],[width-1,i+2],[width-2,i+1]]
	# for j in range(width-2):
	# 	imEdge[(j+1,0)]=[[j,0],[j+2,0],[j+1,1]]
	# 	imEdge[(j+1,height-1)]=[[j,height-1],[j+2,height-1],[j+1,height-2]]
	# for i in range(height-2):
	# 	for j in range(width-2):
	# 		imEdge[(i+1,j+1)]=[[i,j+1],[i+1,j],[i+2,j+1],[i+1,j+2]]
	# return data,imEdge

#dD: dis dictionary
#disList: dis List
# label: dis number
# coe: Lamda


def energyTotal(disList,l1,r1,label,edge,dDict,coe):
	total=0
	for i in disList:
		D=energyData(i[0],i[1],label,l1,r1)
		V=energySmoothness(i[0],i[1],edge,dDict)
		total=D+coe*V
	return total
def energyData(x,y,label,l1,r1):
	h,w=l1.shape
	#print(h,w)
	# print('h = ',h)
	if (y+label+1>=w):
		# deal with boundary of the image. some pixel in right omage do not appear in left image
		return np.absolute(r1[x][y]+1)
	else:
		
		return np.absolute(r1[x][y]-l1[x][y+label+1])
def energySmoothness(x,y,edge,dDict):
	totalcount=0
	A=dDict[(x,y)]
	B=edge[(x,y)] 
	for i in B:
		w=dDict[(i[0],i[1])]
		totalcount=totalcount+np.absolute(w-A)
	return totalcount
def initState(l1,r1,disInd):
	dLL = [[] for x in range(disInd)]
	h,w=r1.shape
	dDict={}
	#sadasd
	for i in range(h):
		for j in range(w):
			helperList=[]
			for d in range(disInd):
				A=energyData(i,j,d,l1,r1)
				# print("A is ", A)
				helperList.append(A)
			# print(helperList)
			list_min = min(helperList)
			helper1=helperList.index(list_min)
			dLL[helper1].append((i,j))
			dDict[(i,j)]=helper1

	return dLL,dDict

def permute(nums):
	result=[]
	for i in permutations(nums,2):
		result.append(list(i))
	print(result)
	return result

def swap(dDict,dLL,edge,l1,r1,disInd):
	counter=0
	helper1=[x for x in range(disInd)]
	helper2=permute(helper1)
	success=0
	while (success == 0):
		for x in helper2:
			temp_energy = []
			temp_alpha = copy.deepcopy(dLL[x[0]])
			temp_beta = copy.deepcopy(dLL[x[1]])
			#temp_beta = dLL[x[1]]
			coe=5
			org_energy = energyTotal(temp_alpha,l1,r1,x[0],edge,dDict,coe)
			org_energy1 = energyTotal(temp_beta,l1,r1,x[1],edge,dDict,coe)
			org_total = org_energy+org_energy1
			for i in dLL[x[0]]:
				temp_alpha.remove(i)
				temp_beta.append(i)
				temp_energy.append(energyTotal(temp_alpha,l1,r1,x[0],edge,dDict,coe)+energyTotal(temp_beta,l1,r1,x[1],edge,dDict,coe))
				print("K")
				temp_alpha.append(i)
				temp_beta.remove(i)
				counter+=1
			print(temp_energy)
			if (len(dLL[x[0]])==0):
				print("not suitable disparity index")
				success=2
				break
			helper3=min(temp_energy)
			print(helper3)
			if (helper3<org_total):
				success=1
				dLL[x[1]].append(dLL[x[0]][temp_energy.index(helper3)])
				del dLL[x[0]][temp_energy.index(helper3)]
				dDict.update({dLL[x[1]][-1]:x[1]})
		if  (success == 1):
			success = 0
		else:
			return dLL,dDict

def main():
	# left_image = imageProcess('image_left.png')
	# right_image = imageProcess('image_right.png')
	# left_image = cv2.imread('image_left.ppm')
	# im=Image.open('image_left.ppm')
	# width,height = im.size
	# pixelNum=width*height
	# im=im.convert("L")
	# data=im.getdata()
	# data=np.matrix(data,dtype='float')
	# new_data=np.reshape(data,(height,width))
	# print(left_image[1])

	# cv2.imshow("first try",left_image)
	# cv2.waitKey(100);
	# print(new_data)

	left_image = imageProcess('test1.png')
	right_image = imageProcess('test2.png')
	# left_pixel = cv2.imread('image_left.png')
	# right_pixel = cv2.imread('image_right.png')
	# print(type(right_image[0][0][0]))
	disInd = 2
	initial= initState(left_image[0],right_image[0],disInd)
	#print(initial[0][0][218])
	A,B=swap(initial[1],initial[0],left_image[1],left_image[0],right_image[0],disInd)
	#initial,dirct=imageProcess("template_K.png")
	#print(dirct[(0,0)])
	#print(dirct[(1,0)])
	#print(dirct[(25,25)])



if __name__ =="__main__":
	start_time=time.time()
	main()
	print("---%s seconds---" %(time.time()-start_time))
