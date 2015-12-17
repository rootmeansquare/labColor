import cv2 
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/home/imaad/Downloads/8.jpg')

image = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)

lChl, aChl, bChl = cv2.split(image)

eq = cv2.createCLAHE()
eq.setClipLimit(1)

aChl =eq.apply(aChl)
bChl =eq.apply(bChl)

eq.setClipLimit(0.5)
lChl = eq.apply(lChl)

'''

aChl = aChl/(np.linalg.norm(aChl))*100*127

bChl = bChl/(np.linalg.norm(bChl))*100*127

aChl = aChl.astype(np.uint8)
bChl = bChl.astype(np.uint8)
'''
result = cv2.merge((lChl,aChl,bChl))

result = cv2.cvtColor(result,cv2.COLOR_LAB2RGB)

cv2.imwrite('/home/imaad/Downloads/result.jpg',result)

plt.figure('')
plt.imshow(image)
plt.imshow(result)