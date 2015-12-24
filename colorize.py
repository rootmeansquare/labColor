import cv2 
import numpy as np
import matplotlib.pyplot as plt
import glob

#Put this script in folder with two direcrtories inside: toColorize, colorized
#toColorized contains images that are to be enhanced
#colorized are the resulting images 

def expandColors(image):

	lChl, aChl, bChl = cv2.split(image)

	eq = cv2.createCLAHE()
	eq.setClipLimit(0.7)

	aChl =eq.apply(aChl)
	bChl =eq.apply(bChl)

	eq.setClipLimit(0.5)
	lChl = eq.apply(lChl)

	result = cv2.merge((lChl,aChl,bChl))

	result = cv2.cvtColor(result,cv2.COLOR_LAB2BGR)

	return (result)

imList = glob.glob('toColorize/*')

for idx,i in enumerate(imList):
	image = cv2.imread(i)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
	colorizedImage = expandColors(image)
	cv2.imwrite('colorized/' +str(idx) + '.jpg',colorizedImage)