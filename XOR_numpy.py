import numpy as np

def sigmoid(x):
   return 1/(1+np.exp(-x))

def dsigmoid(x):
   return x*(1-x)

#input
x = np.array([[0,0],[0,1],[1,0],[1,1]])
#output
y = np.array([[0,1,1,0]]).T


#nn with 3 neuron in hidden layer
#randomly initialising weights
W0 = 2*np.random.random((2,3))-1
W1 = 2*np.random.random((3,1))-1

epoch = 100000

for iter in range(epoch):
   a1 = x
   z1 = np.dot(a1,W0)
   a2 = sigmoid(z1)
   z2 = np.dot(a2,W1)
   a3 = sigmoid(z2)
   a3_error = y-a3
   #backpropagating error
   a3_delta = a3_error*dsigmoid(a3)
   a2_error = np.dot(a3_delta,W1.T)
   a2_delta = a2_error*dsigmoid(a2)
   W1 += 2*np.dot(a2.T,a3_delta)
   W0 += 2*np.dot(a1.T,a2_delta)
   print("Epoch:",iter+1,'error:', np.mean(np.abs(a3_error)))
   
print('After training')
Y = []
for row in a3:
	if row[0] >= 0.90:
		Y.append(1)
	else:
		Y.append(0)
#trained output after last epoch
print (Y)
