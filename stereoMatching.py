# width(y) and height(x) is opposite
import numpy as np
# import PIL.image 	#require PIL package installation. 
import cv2 	#
# from energy import energy.py
from itertools import permutations, combinations # for choosing alpha/beta randomly
import copy # to do deep copy
import time
import maxflow
def imageProcess(file):
	im=cv2.imread(file)
	im2=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	# data=im.getdata()
	data=np.array(im2,dtype='int')
	# new_data=np.reshape(data,(height,width))
	# use directory store neighbors of each vertex
	del im
	del im2

	return data

#	imEdge = {}
#	imEdge[(0,0)]=[[1,0],[0,1]]

	##new version
#	imEdge[(height-1,width-1)]=[[height-1,width-2],[height-2,width-1]]
#	imEdge[(height-1,0)]=[[height-1,1],[height-2,0]]
#	imEdge[(0,width-1)]=[[1,width-1],[0,width-2]]
	###
#	for i in range(height-2):
#		imEdge[(i+1,0)]=[[i,0],[i+2,0],[i+1,1]]
#		imEdge[(i+1,width-1)]=[[i,width-1],[i+2,width-1],[i+1,width-2]]
#	for j in range(width-2):
#		imEdge[(0,j+1)]=[[0,j],[0,j+2],[1,j+1]]
#		imEdge[(height-1,j+1)]=[[height-1,j],[height-1,j+2],[height-2,j+1]]
#	for i in range(height-2):
#		for j in range(width-2):
#			imEdge[(i+1,j+1)]=[[i+1,j],[i,j+1],[i+1,j+2],[i+2,j+1]]

#dD: dis dictionary
#disList: dis List
# label: dis number
# coe: Lamda


def energyTotal(disList,l1,r1,label,dDict,coe1=10,coe2=15,coe3=5):
	total=0
	for i in disList:
		D=energyData(i[0],i[1],label,l1,r1)
		V=energySmoothness(i[0],i[1],r1,dDict,coe1,coe2,coe3)
		total+=(D+V)
	return total
def energyData(x,y,label,l1,r1):
	h,w=l1.shape

	if (y+label+1>=w):
		# deal with boundary of the image. some pixel in right omage do not appear in left image
		# return np.absolute(r1[x][y]-l1[x][y-label-1])
		return energyData(x,y-1,label,l1,r1)
	else:
		
		return np.absolute(r1[x][y]-l1[x][y+label+1])
def energySmoothness(x,y,r1,dDict,coe1=10,coe2=15,coe3=5):
	totalcount=0
	alpha = dDict[x][y]
	h,w=r1.shape
	beta=0
	if(x>=1):
		beta=dDict[x-1][y]
		totalcount+=((alpha-beta)!=0)*(coe1*(np.absolute(r1[x][y]-r1[x-1][y])>=coe3)+coe2*(np.absolute(r1[x][y]-r1[x-1][y])<coe3))
	if(y>=1):
		beta=dDict[x][y-1]
		totalcount+=((alpha-beta)!=0)*(coe1*(np.absolute(r1[x][y]-r1[x][y-1])>=coe3)+coe2*(np.absolute(r1[x][y]-r1[x][y-1])<coe3))
	
	return totalcount

#need all neighbor's energy to decide the energy of s-e-t
def energysmooth(x,y,r1,dDict,beta,coe1=10,coe2=15,coe3=5):
	energeycount=0
	h,w=r1.shape
	alpha=dDict[x][y]
	if(x>=1 and dDict[x-1][y]!=alpha and dDict[x-1][y]!=beta):
		#beta=dDict[x-1][y]
		energeycount+=((alpha-dDict[x-1][y])!=0)*(coe1*(np.absolute(r1[x][y]-r1[x-1][y])>=coe3)+coe2*(np.absolute(r1[x][y]-r1[x-1][y])<coe3))
	if(y>=1 and dDict[x][y-1]!=alpha and dDict[x][y-1]!=beta):
		#beta=dDict[x][y-1]
		energeycount+=((alpha-dDict[x][y-1])!=0)*(coe1*(np.absolute(r1[x][y]-r1[x][y-1])>=coe3)+coe2*(np.absolute(r1[x][y]-r1[x][y-1])<coe3))
	if(x<=h-2 and dDict[x+1][y]!=alpha and dDict[x+1][y]!=beta):
		#beta=dDict[x+1][y]
		energeycount+=((alpha-dDict[x+1][y])!=0)*(coe1*(np.absolute(r1[x][y]-r1[x+1][y])>=coe3)+coe2*(np.absolute(r1[x][y]-r1[x+1][y])<coe3))
	if(y<=w-2 and dDict[x][y+1]!=alpha and dDict[x][y+1]!=beta):
		#beta=dDict[x][y+1]
		energeycount+=((alpha-dDict[x][y+1])!=0)*(coe1*(np.absolute(r1[x][y]-r1[x][y+1])>=coe3)+coe2*(np.absolute(r1[x][y]-r1[x][y+1])<coe3))

	del alpha 
	
	return energeycount




def edgeEnergy(alpha,beta,int1,int2,coe1=10,coe2=15,coe3=5):
	energy = ((alpha-beta)!=0)*((np.absolute(int1-int2)>=coe3)*coe1+coe2*(np.absolute(int1-int2)<coe3))
	return energy


	# alpha = dDict[x1][y1]
	# beta = dDict[x2][y2]
	# count=((alpha-beta)!=0)*(0.3*(np.absolute(r1[x1][y1]-r1[x2][y2])>10)+20*(np.absolute(r1[x1][y1]-r1[x2][y2])<10))
	# return count

def initState(l1,r1,disInd):
	dLL = [[] for x in range(disInd)]
	h,w=r1.shape
	dDict=np.zeros((h,w),dtype=int)
	#sadasd
	hd=0
	for i in range(h):
		for j in range(w):
			M=10000000
			for d in range(disInd):
				A=energyData(i,j,d,l1,r1)
				if(A<M):
					M=A
					hd=d
			dLL[hd].append((i,j))
			dDict[i][j]=hd

	return dLL,dDict

def permute(nums):
	result=list(combinations(nums,2))
	return result

def makeGraph(dDict,dLL1,dLL2,alpha,beta,r1,l1,coe1=10,coe2=15,coe3=5):
# create new graph with vertex in alpha and beta 
# giving energy and cap where cap=energy
###change
	numOfPix=len(dLL1)+len(dLL2)
	newGraph = maxflow.Graph[float](numOfPix,4*numOfPix)
	h,w=r1.shape
	#first para is num of nodes, Second para is num of Edges not accurate number
	nodes=newGraph.add_nodes(numOfPix)
	# return identifiers of node added
	helpDict={}

	for i in range(numOfPix):
		if (i<len(dLL1)):
			helpDict.update({dLL1[i]:i})
		else:
			A=i-len(dLL1)
			helpDict.update({dLL2[A]:i})
	for i in range(numOfPix):
		if (i<len(dLL1)):
			x,y=dLL1[i]
		else :
			x,y=dLL2[i-len(dLL1)]
			
		if ((x+1)<h ):
			neighbor = helpDict.get((x+1,y))
			if (neighbor != None):
				eE=edgeEnergy(alpha, beta,r1[x][y],r1[x+1][y],coe1,coe2,coe3)
				newGraph.add_edge(nodes[i],nodes[neighbor],eE,eE)
			# and (((x,y+1) in dLL1) or ((x,y+1) in dLL2))
		if ((y+1)<w ):
			neighbor1 = helpDict.get((x,y+1))
			if (neighbor1 != None):
				eE1=edgeEnergy(alpha, beta,r1[x][y],r1[x][y+1],coe1,coe2,coe3)
				newGraph.add_edge(nodes[i],nodes[neighbor1],eE1,eE1)

		# sC= energysmooth(x,y,r1,dDict,beta,coe1,coe2,coe3) + energyData(x,y,alpha,l1,r1)
		# tC= energysmooth(x,y,r1,dDict,beta,coe1,coe2,coe3) + energyData(x,y,beta,l1,r1)
		sC= energySmoothness(x,y,r1,dDict,coe1,coe2,coe3) + energyData(x,y,alpha,l1,r1)
		tC= energySmoothness(x,y,r1,dDict,coe1,coe2,coe3) + energyData(x,y,beta,l1,r1)
		newGraph.add_tedge(nodes[i],sC,tC)
	del helpDict
	del numOfPix
	del h
	del w

	return newGraph,nodes

def change_label(alpha,beta,nodes,dLL,newGraph,dDict):
	flow=newGraph.maxflow()
	new_dLL=dLL.copy()
	A=len(dLL[alpha])
	if (dLL[alpha]!=[] or dLL[beta]!=[]):
		new_dLL[alpha]=[]
		new_dLL[beta]=[]
	else:
		return dLL
	for i in nodes:
		node_label = newGraph.get_segment(i)
		#Ind = nodes.where(i)
		if (node_label==1):
			if (i<len(dLL[alpha])):
				new_dLL[alpha].append(dLL[alpha][i])
				dDict[dLL[alpha][i]] = alpha
			else:
				new_dLL[alpha].append(dLL[beta][i-A])
				dDict[dLL[beta][i-A]] = alpha
		else :
			if (i<len(dLL[alpha])):
				new_dLL[beta].append(dLL[alpha][i])
				dDict[dLL[alpha][i]] = beta
			else:
				new_dLL[beta].append(dLL[beta][i-A])
				dDict[dLL[beta][i-A]] = beta

	del flow
	del A

	#not sure if we need change edge relationship
	return new_dLL

def swap(dDict,dLL,l1,r1,disInd,coe1=10,coe2=15,coe3=5):
	counter = 0
	helper1 = [x for x in range(disInd)]
	dis_image = np.zeros(r1.shape,dtype = int)
	helper2 = permute(helper1)
	step = np.floor(255/disInd)
	print(helper2)
	success = 0
	totalEnergy = 0
	iteration = 1
	# finalL=[]
	for y in range(len(dLL)):
		totalEnergy+=energyTotal(dLL[y],l1,r1,y,dDict,coe1,coe2,coe3)
	h,w = r1.shape
	while (success == 0):
		print('Iteration:', iteration, 'Starts!!!')
		for x in helper2:
			newEnergy=0
			newGraph,nodes=makeGraph(dDict,dLL[x[0]],dLL[x[1]],x[0],x[1],r1,l1,coe1,coe2,coe3)
			print('Current alpha-beta is:',x)
			new_dLL=change_label(x[0],x[1],nodes,dLL,newGraph,dDict)
			newGraph,nodes=makeGraph(dDict,dLL[x[0]],dLL[x[1]],x[0],x[1],r1,l1,coe1,coe2,coe3)
			new_dLL=change_label(x[0],x[1],nodes,dLL,newGraph,dDict)


			for z in range(len(new_dLL)):
				newEnergy+=energyTotal(new_dLL[z],l1,r1,z,dDict,coe1,coe2,coe3)
			if (newEnergy < totalEnergy):
				print("New Min Energy Found\n",totalEnergy,newEnergy)
				totalEnergy=newEnergy
				dLL=new_dLL.copy()
				success=1
			del newGraph
			del nodes
			del new_dLL
		current_dis = 0
		for i in dLL:
			for j in i:
				dis_image[j] = current_dis*step
			current_dis += 1
		print('Iteration ',iteration,' Done')
		print('Now Saving Results')
		cv2.imwrite('Result_Iteration'+str(iteration)+'.png',dis_image)
		print('Saving Done!','\n Preparing for Next Iteration\n.\n.\n.')
		iteration += 1



		if  (success == 1):
			success = 0
			# finalL.append(dLL)
		else:
			return dis_image

def main():
	img1=str(input("input left image \n"))
	img2=str(input("input right image \n"))
	if ((len(img1)==0) or (len(img2)==0)):
		print("use default test")
		left_image = imageProcess('scene1.row3.col1.ppm')
		right_image = imageProcess('scene1.row3.col2.ppm')
		disInd = 14
		initial= initState(left_image,right_image,disInd)
		print("Initial State Done!\n Now Swap")
		result_image = swap(initial[1],initial[0],left_image,right_image,disInd)
	else:
	# dis_image = np.zeros(right_image.shape,dtype = int)
	# cv2.imwrite('Initial.png',dis_image)
		left_image = imageProcess(img1)
		right_image = imageProcess(img2)
		disInd = int(input("input how many disparity set you want: \n"))
		coe3 = float(input("threshold of indensity difference: \n"))
		coe1 = float(input("what coefficient you want when intensity difference bigger than threshold: \n"))
		coe2 = float(input("what coefficient you want when intensity difference smaller than threshold: \n"))
		initial= initState(left_image,right_image,disInd)
		print("Initial State Done!\n Now Swap")
		result_image = swap(initial[1],initial[0],left_image,right_image,disInd,coe1,coe2,coe3)

#	result_median_image = cv2.medianBlur(result_image,3)
#	cv2.imwrite('Final Result with Median Filter.png',result_median_image)




if __name__ =="__main__":
	start_time=time.time()
	main()
	print("---%s seconds---" %(time.time()-start_time))
