from sklearn.datasets import load_digits, load_iris
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np

class GMM_EM():
    def __init__(self,n_components,max_iter = 1000,error = 1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.error = error
        self.samples = 0
        self.features = 0
        self.alpha = []
        self.mu = []
        self.sigma = []
    
    def _init(self,data):
        np.random.seed(7)
        self.mu = np.array(np.random.rand(self.n_components, self.features))
        self.sigma = np.array([np.eye(self.features)/self.features] * self.n_components)
        self.alpha = np.array([1.0 / self.n_components] * self.n_components)
        print(self.alpha.shape,self.mu.shape,self.sigma.shape)
            
    def gauss(self, Y, mu, sigma):
        return multivariate_normal(mean=mu,cov=sigma+1e-7*np.eye(self.features)).pdf(Y)
    
    def preprocess(self,data):
        self.samples = data.shape[0]
        self.features = data.shape[1]
        pre = preprocessing.MinMaxScaler()
        return pre.fit_transform(data)
    
    def fit_predict(self,data):
        data = self.preprocess(data)
        self._init(data)
        
        weighted_probs = np.zeros((self.samples,self.n_components))
        for i in range(self.max_iter):
            prev_weighted_probs = weighted_probs
            
            weighted_probs = self._e_step(data)
            change = np.linalg.norm(weighted_probs - prev_weighted_probs) 
            if change < self.error:
                break
            
            self._m_step(data,weighted_probs)
        
        return weighted_probs.argmax(axis = 1)
    
    def _e_step(self,data):
        probs = np.zeros((self.samples,self.n_components))
        for i in range(self.n_components):
            probs[:,i] = self.gauss(data, self.mu[i,:], self.sigma[i,:,:])
        
        weighted_probs = np.zeros(probs.shape)
        for i in range(self.n_components):
            weighted_probs[:,i] = self.alpha[i]*probs[:,i]
            
        for i in range(self.samples):
            weighted_probs[i,:] /= np.sum(weighted_probs[i,:])
        
        return weighted_probs
            
    def _m_step(self,data,weighted_probs):
        for i in range(self.n_components):
            sum_probs_i = np.sum(weighted_probs[:,i])
            
            self.mu[i,:] = np.sum(np.multiply(data, np.mat(weighted_probs[:, i]).T), axis=0) / sum_probs_i
            self.sigma[i,:,:] = (data - self.mu[i,:]).T * np.multiply((data - self.mu[i,:]), np.mat(weighted_probs[:, i]).T) / sum_probs_i
            self.alpha[i] = sum_probs_i/data.shape[0]
    
    def predict_prob(self,data):
        return self._e_step(data)
    
    def predict(self,data):
        return self._e_step(data).argmax(axis = 1)
        
            
dataset = load_iris()
data = dataset['data']
label = dataset['target']

gmm = GMM_EM(3)
pre_label = gmm.fit_predict(data)
print(accuracy_score(pre_label,label))




















