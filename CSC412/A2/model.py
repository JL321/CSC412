import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle 
from matplotlib.patches import Ellipse
#zs is of size KxN
#data is of size Mx2

def log_prior(zs):	

    log_p = -0.5*(np.log(2*np.pi)+torch.square(zs))
    return torch.sum(log_p, axis=1)

def logp_a_beats_b(za, zb):

    return -torch.log1p(torch.exp(zb-za))

def all_games_log_likelihood(zs, games):

    '''
    result = []
    for i in range(zs.shape[0]):
            result.append(np.squeeze(np.sum(logp_a_beats_b(zs[i][games[:][0]], zs[i][games[:][0]]))))
    result = np.array(result)
    ''' 
    result = torch.sum(logp_a_beats_b(torch.transpose(zs, 0, 1)[torch.from_numpy(games[0]).long()], torch.transpose(zs, 0, 1)[torch.from_numpy(games[1]).long()]), axis=0)
    #result = torch.transpose(result, 0, 1)  #Now Kx1

    return result

def joint_log_density(zs, games):
    '''
    print("Log Likelihood:")
    print(all_games_log_likelihood(zs, games))
    print("Log Prior:")
    print(log_prior(zs))
    '''
    return log_prior(zs)+all_games_log_likelihood(zs, games)

def mnorm_pdf(mu, var, x):
    cov = var*np.eye(var.shape[-1])
    return np.squeeze(-0.5*(np.log(2*np.pi) + np.log(np.prod(var)) + np.transpose(x-mu) @ (1/cov) @ (x-mu)))

def mnorm_pdf_torch(mu, var, x):
    #  Returns the log pdf for variational distribution
    return torch.sum(-0.5*(np.log(2*np.pi) + torch.log(var) + torch.square(x-mu)/var), axis=1)

def elbo(params, logp, num_samples, data):

    #Assuming params are stored in a 2xN format (for N agents)
    #Assuming mean field variational inference, no direct dependence on D for approximate distribution
    #Assuming logp refers to a function computing the joint posterior over data and latent variables

    mu, var = params
    zs_sample = np.random.normal(size=(num_samples, 2)) #Assuming 2 agents
    zs_sample = torch.from_numpy(zs_sample).double()
    zs = zs_sample*var+mu
    elbo = torch.mean(logp(zs, data)-mnorm_pdf_torch(mu, var, zs))
    #print("ELBO: {}".format(elbo))
    return elbo

def neg_toy_elbo(params, data):
    return -elbo(params, joint_log_density, 100, data)

def fit_toy_variational_dist(params, data, iters=10000, lr=1e-4):
    data = np.transpose(data)
    mu, var = params
    for i in range(iters): 
        elbo_loss = neg_toy_elbo(params, data)
        #print(elbo_loss)   
        if (i+1)%100 == 0:
            print("Loss: {}, Step: {}".format(elbo_loss, i+1))
        elbo_loss.backward()
        mu.data -= lr*mu.grad
        var.data -= lr*var.grad
        mu.grad.zero_()
        var.grad.zero_()		
    return mu, var

if __name__ == '__main__':

    train = True
    data = np.array([[0, 1]])
    mu = torch.tensor(np.random.normal(size=2), dtype=torch.double, requires_grad=True)
    var = torch.tensor(np.random.randint(1, 2, 2), dtype=torch.double, requires_grad=True)
    if train:
        trained_mu, trained_var = fit_toy_variational_dist((mu, var), data)
        trained_mu = np.squeeze(trained_mu.detach().numpy())
        trained_var = np.squeeze(trained_var.detach().numpy())
        pickle.dump([trained_mu, trained_var], open("qparam.p", "wb"))
    else:
        trained_mu, trained_var = pickle.load(open("qparam.p", "rb"))
    
    covariance = np.eye(2)*trained_var
    eigval, eigvec = np.linalg.eig(covariance)
    print(eigval)
    pvec = np.argmax(eigval)
    rotation_angle = np.arctan(eigvec[pvec][1]/(eigvec[pvec][0]+1e5))
    print("Eigen Vectors: {}".format(eigvec))
    print(covariance)
    print("Player A mu: {} Player B mu: {}".format(trained_mu[0], trained_mu[1]))
    fig, ax = plt.subplots()
    samples = np.random.multivariate_normal(trained_mu, covariance, size=1000)
    ellipse = Ellipse((trained_mu[0], trained_mu[1]), eigval[pvec]*2, eigval[pvec-1]*2, angle=rotation_angle*180/np.pi, edgecolor='red', facecolor='none')
    ax.add_patch(ellipse) 
    ax.scatter(np.transpose(samples)[0], np.transpose(samples)[1], c='blue', s=2)  
    ax.set_xlabel("Player A")
    ax.set_ylabel("Player B")
    ax.set_title("Distribution of latent variables between A and B")
    plt.show()
