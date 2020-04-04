# width(y) and height(x) is opposite
from matplotlib import pyplot
import numpy as np
from PIL import Image
# import PIL.image 	#require PIL package installation. 
import cv2 	#
# from energy import energy.py
import collections # check 2 list has same elements elements 
from itertools import permutations, combinations # for choosing alpha/beta randomly
import copy # to do deep copy
import time
import maxflow
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
def edgeEnergy(alpha,beta,x1,y1,x2,y2,r1):
	count=np.absolute(alpha-beta)*20*(np.absolute(r1[x1][y1]-r1[x2][y2]))
	return count

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
	result=list(combinations(nums,2))
	print(result)
	return result

def makeGraph(dDict,dLL,edge,alpha,beta,r1,l1):
# create new graph with vertex in alpha and beta 
# giving energy and cap where cap=energy
	pixInA=dLL[alpha]
	pixInB=dLL[beta]
	pixInA=pixInA.extend(pixInB)
	numOfPix=len(pixInA)
	print(numOfPix)
	newGraph = maxflow.Graph[float](numOfPix,4*numOfPix)
	#first para is num of nodes, Second para is num of Edges not accurate number
	nodes=g.add_nodes(numOfPix)
	# return identifiers of node added
	for i in pixInA:
		iInd=pixInA.index(i)
		if(i[0]>=0):
			if ((i[0]+1,i[1]) in dDict.keys()):
				neighbor = pixInA.index((i[0]+1,i[1]))
				eE=edgeEnergy(alpha,beta,i[0],i[1],i[0]+1,i[1],r1)
				newGraph.add_edge(nodes[iInd],nodes[neighbor],eE,eE)
		if(i[1]>=0):
			if ((i[0],i[1]+1) in dDict.keys()):
				neighbor1 = pixInA.index((i[0],i[1]+1))
				eE1=edgeEnergy(alpha,beta,i[0],i[1],i[0],i[1]+1,r1)
				newGraph.add_edge(nodes[iInd],nodes[neighbor1],eE1,eE1)
		sC= energySmoothness(i[0],i[1],edge,dDict) + energyData(i[0],i[1],alpha,l1,r1)
		tC= energySmoothness(i[0],i[1],edge,dDict) + energyData(i[0],i[1],beta,l1,r1)
		newGraph.add_tedge(nodes[iInd],sC,tC)
	return pixInA,newGraph,nodes

def change_label(alpha,beta,pixInA,nodes,dLL,edge,newGraph):
	for i in nodes:
		node_label = newGraph.get_segment(i)
		Ind = nodes.index(i)
		new_dLL=dLL.copy()
		new_dLL[alpha]=[]
		new_dLL[beta]=[]
		if (node_label):
			new_dLL[alpha].append(pixInA[Ind])
		else :
			new_dLL[beta].append(pixInA[Ind])
	#not sure if we need change edge relationship
	return new_dLL,edge

def swap(dDict,dLL,edge,l1,r1,disInd):
	counter=0
	helper1=[x for x in range(disInd)]
	helper2=permute(helper1)
	success=0
	totalEnergy=0
	for y in dLL:
		totalEnergy+=energyTotal(y,l1,r1,dLL.index(y),edge,dDict,coe)
	h,w = r1.shape
	coe=15
	while (success == 0):
		for x in helper2:
			newEnergy=0
			pixInA,newGraph,nodes=makeGraph(dDict,dLL,edge,x[0],x[1],r1,l1)
			new_dLL=change_label(x[0],x[1],pixInA,nodes,dLL,edge,newGraph)
			for z in new_dLL:
				newEnergy+=energyTotal(z,l1,r1,new_dLL.index(z),edge,dDict,coe)
			if (newEnergy < totalEnergy):
				totalEnergy=newEnergy
				dLL=new_dLL.copy()
				success=1
		if  (success == 1):
			success = 0
		else:
			return dLL,dDict

def main():
	left_image = imageProcess('image_left.png')
	right_image = imageProcess('image_right.png')
	disInd = 2
	initial= initState(left_image[0],right_image[0],disInd)
	A,B=swap(initial[1],initial[0],left_image[1],left_image[0],right_image[0],disInd)
	# show disparity image
	dis_im = np.uint8(np.zeros((right_image[0].shape[0],right_image[0].shape[1]),dtype = int))
	# dis_im = np.zeros((right_image[0].shape[0],right_image[0].shape[1]),dtype = int)
	dis_im += 255
	for i in range(disInd):
		for j in A[i]:
			dis_im[j[0]][j[1]] -=  i*50
	
	cv2.imshow('test',dis_im)
	cv2. waitKey(10000)


if __name__ =="__main__":
	start_time=time.time()
	main()
	print("---%s seconds---" %(time.time()-start_time))
