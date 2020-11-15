import numpy as np
import matplotlib.pyplot as plt
import torch

#Exercise 2.2

def target_f1(x, s_true=0.3):
    noise = np.random.rand(*x.shape)
    y = 2*x+s_true*noise
    return y

def target_f2(x):
    noise = np.random.rand(*x.shape)
    y = 2*x+np.linalg.norm(x)*0.3*noise
    return y

def target_f3(x):
    noise = np.random.rand(*x.shape)
    y = 2*x+5*np.sin(0.5*x)+np.linalg.norm(x)+0.3*noise
    return y

def sample_batch(target_f, batch_size):
    x = np.random.uniform(0, 20, [1, batch_size])
    y = target_f(x)
    y = np.reshape(y, (-1, 1))
    return (x, y)

plt.xlabel("Inputs Samples")
plt.ylabel("Target Samples")
x1, y1 = sample_batch(target_f1, 1000)
plt.scatter(x1, y1, 1, c="blue")

x2, y2 = sample_batch(target_f2, 1000)
plt.scatter(x2, y2, 1, c="red")

x3, y3 = sample_batch(target_f3, 1000)
plt.scatter(x3, y3, 1, c="green")

#Exercise 2.3

def beta_mle(X, Y):
    print("Verify X, Y Shape: {}, {}".format(X.shape, Y.shape))
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X, np.transpose(X))), X), Y)
    return beta

def convert_matrix(x):
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, 0)
    #print("Verify Shape: {}".format(x.shape))
    return x

b_mle_1 = beta_mle(x1, y1)
b_mle_2 = beta_mle(x2, y2)
b_mle_3 = beta_mle(x3, y3)

print(b_mle_1.shape)
regression_1 = [np.random.multivariate_normal(np.sum(np.matmul(np.transpose(convert_matrix(v)), b_mle_1), axis=1), np.eye(1,1)) for v in x1[0]]
regression_2 = [np.random.multivariate_normal(np.sum(np.matmul(np.transpose(convert_matrix(v)), b_mle_2), axis=1), np.eye(1,1)) for v in x2[0]]
regression_3 = [np.random.multivariate_normal(np.sum(np.matmul(np.transpose(convert_matrix(v)), b_mle_3), axis=1), np.eye(1,1)) for v in x3[0]]

regression_1 = np.squeeze(regression_1)
regression_2 = np.squeeze(regression_2)
regression_3 = np.squeeze(regression_3)

plt.scatter(x3, regression_3, 1, c="pink")
x1 = np.squeeze(x1)
x2 = np.squeeze(x2)
x3 = np.squeeze(x3)

m1, b1 = np.polyfit(x1, regression_1, 1)
m2, b2 = np.polyfit(x2, regression_2, 1)
m3, b3 = np.polyfit(x3, regression_3, 1)

pred_y1 = np.squeeze(m1*x1+b1)
pred_y2 = np.squeeze(m2*x2+b2)
pred_y3 = np.squeeze(m3*x3+b3)

plt.plot(x1, pred_y1, c = "black")
plt.fill_between(x1, pred_y1-1, pred_y1+1)
plt.plot(x2, pred_y2)
plt.fill_between(x2, pred_y2-1, pred_y2+1)
plt.plot(x3, pred_y3)
plt.fill_between(x3, pred_y3-1, pred_y3+1)

#plt.show()

#2.4

def gaussian_ll(mu, s, x):
    return -0.5*(np.log(2*np.pi)+np.log(s)+np.square(x-mu)/s)

def lr_model_nll(beta, x, y, s=1):
    return np.sum(-gaussian_ll(np.squeeze(np.matmul(np.transpose(x), beta)), np.ones(x.shape[0])*np.square(s), np.squeeze(y)))

def ll_pytorch(mu, s, x):
    x = torch.from_numpy(x).double()
    return -0.5*(np.log(2*np.pi)+torch.log(s)+torch.square(x-mu)/s)
    
def nll_pytorch(beta, x, y, s=1):
    x = torch.from_numpy(x).double()
    return torch.sum(-ll_pytorch(torch.squeeze(torch.transpose(x, 0, 1).mm(beta)), torch.ones(x.shape[0])*np.square(s), np.squeeze(y)))

for n in (10, 100, 1000):
    print("--samplecount_{}--".format(n))
    for target_f in (target_f1, target_f2, target_f3):
        print("--{}--".format(target_f.__name__))
        for s in (0.1, 0.3, 1, 2):
            print("--sigmav_{}--".format(s))
            x, y = sample_batch(target_f, n)
            b_mle = beta_mle(x, y)
            nll = lr_model_nll(b_mle, x, y, s)
            print("nll: {}".format(nll))

#4. f1:s=0.3 f2: 2 f3:2 (n=10)

#2.5 

beta = torch.rand(1, 1, dtype=torch.double, requires_grad=True)
x, y = sample_batch(target_f1, 100)
nll = nll_pytorch(beta, x, y)
nll.backward()
ad_grad = beta.grad
manual_grad = np.squeeze(-np.matmul(x, y) + np.matmul(np.matmul(x, np.transpose(x)), beta.detach().numpy()))
print(ad_grad)
print(manual_grad)

def train_lin_reg(target_f, b_init, bs=100, lr=1e-6, iters=1000, s_model=1):
    print('b_init: {}'.format(b_init))
    print(b_init.is_leaf)
    for _ in range(iters):
        x, y = sample_batch(target_f, bs)
        nll_iter = nll_pytorch(b_init, x, y)
        nll_iter.backward()
        b_init.data -= lr*b_init.grad
        b_init.grad.zero_()
    return b_init

plt.clf()

b_init = torch.tensor(1000*np.random.rand(1, 1), dtype=torch.double, requires_grad=True)
x2_n, y2_n = sample_batch(target_f2, 1000)
b_learned = train_lin_reg(target_f2, b_init, iters=10000).detach().numpy()
y_reg = [np.random.multivariate_normal(np.sum(np.matmul(np.transpose(convert_matrix(v)), b_learned), axis=1), np.eye(1,1)) for v in x2_n[0]]

x2_n = np.squeeze(x2_n)
y_reg = np.squeeze(y_reg)
y2_n = np.squeeze(y2_n)
plt.scatter(x2_n, y2_n, s=1, c='red')
m2_n, b2_n = np.polyfit(x2_n, y_reg, 1)
out = m2_n*x2_n+b2_n 
plt.plot(x2_n, out) 
plt.fill_between(x2_n, out-1, out+1)
plt.show()
