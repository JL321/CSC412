import torch 
import torch.nn as nn
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import torch.optim as optim

def load_mnist(dir_path='/home/jamesl/Downloads/'):
	data, _ = loadlocal_mnist(dir_path+'train-images-idx3-ubyte', dir_path+'train-labels-idx1-ubyte')
	bin_data = binarize_mnist(np.array(data))
	return bin_data

def binarize_mnist(x):
	#  Lazy binarization
	#  Set all pixel values under 21 as 0
	on = x > 20
	x = x*0
	x[on] = 1
	return x

def log_prior(z):
	#  Assuming a standard multivariate normal (dim=2)
	#  Numpy used as prior has no trainable parameters
	return torch.sum(-0.5*(np.log(2*np.pi)+torch.square(z)), axis=1)

def bernoulli_ll(x, logits):
	prob = torch.exp(logits)/(1+torch.exp(logits))
	return torch.sum(torch.log(prob)*x + torch.log(1-prob)*(1-x), axis=1)

def decoder(z, params):
	
	#  Assuming params is a tuple containing matrices of the form (2x500, 500x784)
	w1, b1, w2, b2 = params
	z1 = torch.tanh(torch.mm(z, w1)+b1)
	logits = torch.mm(z1, w2)+b2
	#  Directly compute logits instead of mu (assume relationship can be captured by network) for numerical stability
	return logits

def log_likelihood(z, x, params):
	
	#  Assuming Bx784 data representation
	logits = decoder(z, params)
	ll = bernoulli_ll(x, logits)
	return ll
	
def joint_log_density(z, x, params):
	#print("Log Prior: {}, LL: {}".format(log_prior(z).data, log_likelihood(z, x, params).data))
	return log_prior(z).detach()+log_likelihood(z, x, params)

def encoder(x, params):
	
	#  Assuming param shape of (784x500, 500x4)
	w1, b1, w2, b2 = params
	z1 = torch.tanh(torch.mm(x, w1)+b1)
	z2 = torch.mm(z1, w2)+b2
	mean = z2[:, :2]
	log_std = z2[:, 2:]
	return mean, log_std

def log_q(z, params):
	
	mu, std = params
	#  Assuming param shape of (Bx2, Bx2)
	return torch.sum(-0.5*(np.log(np.pi*2)+2*std+torch.square((z-mu)/torch.exp(std))), axis=1)

def elbo(x, encoder_params, decoder_params, display=False):
	eps = np.random.normal(size=(x.shape[0], 2))
	eps = torch.from_numpy(eps).double()
	mu, log_std = encoder(x, encoder_params)
	reparam_z = eps*torch.exp(log_std)+mu #  Of (Bx2) shape
	#   Assuming pre-mean elbo to be of Bx1 shape
	elbo = torch.mean(joint_log_density(reparam_z, x, decoder_params)-log_q(reparam_z, (mu, log_std)))/784
	if display:
		print("JLL: {}".format(joint_log_density(reparam_z, x, decoder_params)))
		print("Q LL: {}".format(log_q(reparam_z, (mu, log_std))))
	return elbo

def loss(x, encoder_params, decoder_params, display=False):
	return -elbo(x, encoder_params, decoder_params, display)

def plotImage(decoder_params):
	w1, b1, w2, b2 = decoder_params
	x = torch.from_numpy(np.random.normal(size=(1,2))).double()
	z1 = torch.tanh(torch.mm(x, w1))
	logits = torch.mm(z1, w2)
	prob = torch.exp(logits)/(1+torch.exp(logits))
	image = torch.squeeze(prob).detach().view(28, 28).numpy()
	plt.imshow(image)
	plt.show()

def train(x, encoder_params, decoder_params, batch_size=128, epochs=100, lr=1e-3):
	first = True
	x = torch.from_numpy(x).double()
	encoder_l1, encoder_l1b, encoder_l2, encoder_l2b = encoder_params
	decoder_l1, decoder_l1b, decoder_l2, decoder_l2b = decoder_params
	encoder_opt = optim.Adam([encoder_l1, encoder_l1b, encoder_l2, encoder_l2b], lr=1e-4)
	decoder_opt = optim.Adam([decoder_l1, decoder_l1b, decoder_l2, decoder_l2b], lr=1e-4)
	#  Assuming x represents the entire dataset
	data_idx = 0
	access_idx = np.arange(x.shape[0])
	for i in range(epochs):
		while (data_idx < x.shape[0]):
			if data_idx+batch_size >= x.shape[0]:
				end_idx = x.shape[0]
			else:
				end_idx = data_idx+batch_size
			batch_set = x[access_idx[data_idx:end_idx]]
			
			#  Train
			if first:
				vae_loss = loss(batch_set, encoder_params, decoder_params)
				print("First VAE loss")
				print(vae_loss)
				first=False
			else:
				vae_loss = loss(batch_set, encoder_params, decoder_params)
			vae_loss.backward()
			encoder_opt.step()
			decoder_opt.step()
			encoder_opt.zero_grad()
			decoder_opt.zero_grad()
			'''
			encoder_l1.data -= lr*encoder_l1.grad
			encoder_l2.data -= lr*encoder_l2.grad
			decoder_l1.data -= lr*decoder_l1.grad
			decoder_l2.data -= lr*decoder_l2.grad
			encoder_l1.grad.zero_()
			encoder_l2.grad.zero_()
			decoder_l1.grad.zero_()
			decoder_l2.grad.zero_()
			'''
			data_idx += batch_size
		data_idx = 0
		access_idx = np.random.permutation(x.shape[0])
		print("Loss at epoch {}: {}".format(i+1, vae_loss))
		#plotImage(decoder_params)
		
	
	return encoder_params, decoder_params
	
if __name__ == '__main__':
	
	data = load_mnist()
	std = np.sqrt(0.01)
	encoder_l1 = torch.tensor(np.random.normal(scale=std, size=(784, 500)), dtype=torch.double, requires_grad=True)
	encoder_l1b = torch.tensor(np.zeros((1, 500)), dtype=torch.double, requires_grad=True)
	encoder_l2 = torch.tensor(np.random.normal(scale=std, size=(500, 4)), dtype=torch.double, requires_grad=True)
	encoder_l2b = torch.tensor(np.zeros((1,4)), dtype=torch.double, requires_grad=True)
	decoder_l1 = torch.tensor(np.random.normal(scale=std, size=(2, 500)), dtype=torch.double, requires_grad=True)
	decoder_l1b = torch.tensor(np.zeros((1, 500)), dtype=torch.double, requires_grad=True)
	decoder_l2 = torch.tensor(np.random.normal(scale=std, size=(500, 784)), dtype=torch.double, requires_grad=True)
	decoder_l2b = torch.tensor(np.zeros((1,784)), dtype=torch.double, requires_grad=True)
	trained_ep, trained_dp = train(data, (encoder_l1, encoder_l1b, encoder_l2, encoder_l2b), (decoder_l1, decoder_l1b, decoder_l2, decoder_l2b))
	