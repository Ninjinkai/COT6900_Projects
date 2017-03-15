import numpy as np

# Sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# Input data set
X = np.array([ [0,0,1], [0,1,1], [1,0,1], [1,1,1]])
# Output data set
y = np.array([[0,0,1,1]]).T
# Random seed
np.random.seed(1)

# Initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

iterations = 10000
for iter in range(iterations):
#     Forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

#     How much was missed
    l1_error = y - l1

#     Multiply error by slope of sigmoid at values in l1
    l1_delta = l1_error * nonlin(l1, True)

#     Update weights
    syn0 += np.dot(l0.T, l1_delta)

#     Examining l1
    if (iter == 0):
        print('iter = 0')
        print(l1_error, "\n", l1)
    if (iter == iterations/2):
        print('iter = ', iterations/2)
        print(l1_error, "\n", l1)
    if (iter == iterations - 1):
        print('iter = ', iterations - 1)
        print(l1_error, "\n", l1)

print("Output after training:")
print(l1)

# syn0 = 2*np.random.random((3,4)) - 1
# syn1 = 2*np.random.random((4,1)) - 1
#
# for j in range(6000):
#     l1 = 1/(1+np.exp(-(np.dot(x, syn0))))
#     l2 = 1 / (1 + np.exp(-(np.dot(l1, syn1))))
#
#     l2_delta = (y - l2)*(l2*(1-l2))
#     l1_delta = l2_delta.dot(syn1.T)*(l1*(1-l1))
#
#     syn1 += l1.T.dot(l2_delta)
#     syn0 += x.T.dot(l1_delta)