import sys, os
sys.path.append(os.pardir)
import numpy as np
from PIL import Image
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

def img_show(img) : 
	img = Image.fromarray(np.uint8(img))
	img.show()
def img_save(img, name) :
	img = Image.fromarray(np.uint8(img))
	img.save(name + ".png", "PNG")

class Relu:
	def __init__(self):
		self.mask = None
	def forward(self, x):
		self.mask = (x <= 0)
		out = x.copy()
		out[self.mask] = 0
		return out
	def backward(self, dout):
		dout[self.mask] = 0
		dx = dout
		return dx

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


#Dataset load
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, flatten =True, one_hot_label = True)

#Settings
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.00001
batch_mask = batch_mask = np.random.choice(train_size, batch_size)
weight_init_std = 0.01
input_size = x_train[0].shape[0]
hidden_size = 1000
output_size = input_size 



params={}
params['W1'] = weight_init_std * np.random.randn(input_size , hidden_size)
params['b1'] = np.zeros(hidden_size)
params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
params['b2'] = np.zeros(output_size)

layers = {}
layers['Affine1'] = Affine(params['W1'], params['b1'])
layers['Relu'] = Relu()
layers['Affine2'] = Affine(params['W2'], params['b2'])

for i in range(iters_num) : 
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = np.copy(x_train[batch_mask])
	t_batch = np.copy(x_train[batch_mask])

	grads = {}

	a1 = x_batch
	z1 = layers['Affine1'].forward(a1)
	a2 = layers['Relu'].forward(z1)
	z2 = layers['Affine2'].forward(a2)
	#print(z2[0])
	#print(t_batch[0])
	loss = z2 - t_batch
	#print(np.sum(loss**2) / batch_size)
	#print(loss.shape)

	dout = loss
	dout=layers['Affine2'].backward(dout)
	dout=layers['Relu'].backward(dout)
	dout=layers['Affine1'].backward(dout)
	
	"""
	dout = loss 
	for layer in layers :
		dout = layer.backward(dout)
		"""

	grads['W1'] = layers['Affine1'].dW
	grads['b1'] = layers['Affine1'].db
	grads['W2'] = layers['Affine2'].dW
	grads['b2'] = layers['Affine2'].db

	for key in ('W1', 'b1', 'W2', 'b2') :
		params[key] -= learning_rate * grads[key]

	if (i % 100 == 0):
		print(i)
		print(np.sum(loss**2) / batch_size)

for k in range(11):
	a1 = x_train[k]
	#print(a1)
	z1 = layers['Affine1'].forward(a1)
	a2 = layers['Relu'].forward(z1)
	z2 = layers['Affine2'].forward(a2)	
	z2 = z2.reshape(28,28)
	z2 = z2*256


	a1 = (a1*256).reshape(28,28)
	img_save(a1, "input" + str(k))
	img_save(z2, "output" + str(k))

