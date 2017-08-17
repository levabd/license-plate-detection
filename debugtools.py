import cv2
import numpy as np
import mahotas as mh
from pylab import imshow, show

def getAccuracyBCP1(ranges, imagepath, scale):
	imagepath = imagepath.replace('.jpg','M.jpg')
	mask = cv2.imread(imagepath)
	mask = cv2.resize(mask,(0,0),fx=scale,fy=scale)
	
	masksum = float(np.count_nonzero(mask))
	maxAcc = float(0)	

	for r in ranges:
		rsum = np.count_nonzero(mask[r[0]:r[1],:])

		acc = rsum/masksum

		if acc > maxAcc:
			maxAcc = acc

	return maxAcc, mask


def getAccuracy(ranges, mask):
	masksum = float(np.count_nonzero(mask))
	maxAcc = float(0)	

	for r in ranges:
		rsum = np.count_nonzero(mask[r[0]:r[1],r[2]:r[3]])

		acc = rsum/masksum

		if acc > maxAcc:
			maxAcc = acc

	return maxAcc


def getFinalAccuracy(ranges, mask):
	masksum = float(np.count_nonzero(mask))
	maxAcc = float(0)
	maxPer = float(0)

	for r in ranges:
		rsum = np.count_nonzero(mask[r[0]:r[1],r[2]:r[3]])

		acc = rsum/masksum
		per = rsum/((r[1]-r[0])*(r[3]-r[2]))

		if acc * per > maxAcc * maxPer:
			maxAcc = acc
			maxPer = per

	if maxAcc > 0.5 and maxPer > 0.8:
		return True

	return False

def getSegmentation1Accuracy(vLines, mask):
	gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	ret, thres = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
	labeled, nr_objects = mh.label(thres)

	labelAcc = np.zeros(nr_objects)
	labelArea = np.bincount(labeled.flatten())[1:]

	for i in range(len(vLines)-1):
		band = labeled[:,vLines[i]:vLines[i+1]]
		bins = np.bincount(band.flatten())
		if bins.shape[0] > 1:
			maxArea = np.max(bins[1:])
			maxLabel = np.argmax(bins[1:])

			acc = maxArea/float(labelArea[maxLabel])
			if acc > labelAcc[maxLabel]:
				labelAcc[maxLabel] = acc

	return nr_objects, labelAcc[labelAcc > 0.9].shape[0]