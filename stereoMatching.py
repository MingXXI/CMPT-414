from matplotlib import pyplot
import numpy as np
import PIL.Image
import cv2
from energy import energy.py
import collections # check 2 list has same elements elements 
from itertools import permutations # for choosing alpha/beta randomly
import copy 

def imageProcess(file):
	im=Image.open(file)
	width,height = im.size
	pixelNum=width*height
	im=im.convert("L")
	data=im.getdata()
	data=np.matrix(data,dtype='float')
	new_data=np.reshape(data,(height,width))
	# use directory store neighbors of each vertex
	imEdge[(0,0)]=[[2,0],[1,0],[0,1]]
	imEdge[(width-1,height-1)]=[[2,0],[width-2,height-1],[width-1,height-2]]
	imEdge[(0,height-1)]=[[2,0],[1,height-1],[0,height-2]]
	imEdge[(width-1,0)]=[[2,0],[width-1,1],[width-1-2,0]]
	for i in height-3:
		imEdge[(0,i+1)]=[[3,0],[0,i],[0,i+2],[1,i+1]]
		imEdge[(width-1,i+1)]=[[3,0],[width-1,i],[width-1,i+2],[width-2,i+1]]
	for j in width-3:
		imEdge[(j+1,0)]=[[3,0],[j,0],[j+2,0],[j+1,1]]
		imEdge[(j+1,height-1)]=[[3,0],[j,height-1],[j+2,height-1],[j+1,height-2]]
	for i in height-3:
		for j in width-3:
			imEdge[(i+1,j+1)]=[[4,0],[i,j+1],[i+1,j],[i+2,j+1],[i+1,j+2]]
	return new_data,imEdge

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
	w,h=l1.size
	if (x+label+1>=h):
		# deal with boundary of the image. some pixel in right omage do not appear in left image
		return np.absolute(r1[x][y]+1)
	else :
		return np.absolute(r1[x][y]-l1[x+label+1][y])
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
	w,h=r1.size
	dDict={}
	for i in range(w):
		for j in range(h):
			helperList=[]
			for d in range(disInd):
				A=energyData(i,j,d,l1,r1)
				helperList.append(A)
			helper1=helperList.index(min(helperList))
			dLL[helper1].append((i,j))
			dDict[(i,j)]=helper1

	return dLL,dDict

def permute(nums):
	result=[]
	for i in permutations(nums,2):
		result.append(list(i))
	return result

def swap(dDict,dLL,edge,l1,r1,disInd):
	helper1=[x for x in range(disInd)]
	helper2=permute(helper1)
	success=0
	while (success == 0):
		for x in helper2:
			temp_energy = []
			temp_alpha = copy.deepcopy(dLL[x[0]])
			#temp_beta = dLL[x[1]]
			coe=20
			org_energy = energyTotal(temp_alpha,l1,r1,x[0],edge,dDict,coe)
			for i in dLL[x[0]]:
				temp_alpha.remove(i)
				temp_energy.append(energyTotal(temp_alpha,l1,r1,x[0],edge,dDict,coe))
				temp_alpha.append(i)
			helper3=min(temp_energy)
			if (helper3<org_energy):
				success=1
				del dLL[x[0]][temp_energy.index(helper3)]
				dLL[x[1]].append(dLL[x[0]][temp_energy.index(helper3)])
				dDict.update({dLL[x[0]][temp_energy.index(helper3)],x[1]})
		if  (success == 1):
			success = 0
		else:
			return dLL,dDict




if __name__ =="__main__":
	main()
