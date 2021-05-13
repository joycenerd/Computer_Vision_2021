import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse


parser=argparse.ArgumentParser()
parser.add_argument("--img",type=str,default="Mesona",help="Which set of image do you use")
args=parser.parse_args()


DATA_PATH="./data/"


def read_intrinsic():
	if args.img=="Mesona":
		K1=np.array([1.4219,0.0005,0.509,0,1.4219,0.3802,0,0,0.001])
		K2=K1
	elif args.img=="Statue":
		K1=np.array([5426.566895,0.678017,330.096680,0.000000,5423.133301,648.950012,0.000000,0.000000,1.000000])
		K2=np.array([5426.566895,0.678017,387.430023,0.000000,5423.133301,620.616699,0.000000,0.000000,1.000000])
	
	# The last element should be 1 and reshape to (3,3)
	K1/=K1[-1]
	K1=K1.reshape((3,3))
	
	K2/=K2[-1]
	K2=K2.reshape((3,3))

	return K1,K2


if __name__=="__main__":
	img1=cv2.imread(DATA_PATH+'Mesona1.JPG')
	img2=cv2.imread(DATA_PATH+'Mesona2.JPG')
	K1,K2=read_intrinsic()
	print(K1)
	print(K2)