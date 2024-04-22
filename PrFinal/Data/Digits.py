

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.matlab as matlab


# Load the database
mat_file =  "BigDigits.mat"
mat = matlab.loadmat(mat_file,squeeze_me=True) # dictionary
list(mat.keys()) # list vars

taska = True

data = mat["data"]      # read feature vectors
labs = mat["labs"] - 1  # read labels 1..10

allNlabs = np.unique(labs) # all labs 0 .. 9

classsiz = ()
for c in allNlabs:
    classsiz = classsiz + (np.size(np.nonzero(labs==c)),)  
print '\n%% Class labels are: %s' % (allNlabs,)    
print '%% Class frequencies are: %s' % (classsiz,)


# Let's say my digit is ...
myDigit = 7

otherDigits  = np.setdiff1d(allNlabs,myDigit)
other3Digits = np.random.permutation(otherDigits)[:3]

if taska:
    others = other3Digits
else:
    others = otherDigits

print 'class 1 = %s' % myDigit
print 'class 2 = %s' % others

# To construct a 2-class dataset you can use the same matrix
# data and change the vector of labels

aux = labs
classone = np.in1d(labs,myDigit)
classtwo = np.in1d(labs,others)
aux[classone] = 0  # class one
aux[classtwo] = 1  # class two

# Features
X = data[np.logical_or(classone,classtwo)]
# (unchanged) labels
y = aux[np.logical_or(classone,classtwo)]




# Show some digits


hwmny = 20
some1 = np.random.permutation(np.where(y==0)[0])[:hwmny]
some2 = np.random.permutation(np.where(y==1)[0])[:hwmny]



img1 = np.reshape(X[some1,:],(28*hwmny,28)).T
plt.figure(figsize=(10,3))
plt.imshow(img1, cmap=plt.cm.gray_r)
plt.xticks([])
plt.yticks([])
plt.title('Show digits in class one (0) = '+str(myDigit) )
plt.show()


img2 = np.reshape(X[some2,:],(28*hwmny,28)).T
plt.figure(figsize=(10,3))
plt.imshow(img2, cmap=plt.cm.gray_r)
plt.xticks([])
plt.yticks([])
plt.title('Show digits in class two (1) = '+str(others) )
plt.show()





