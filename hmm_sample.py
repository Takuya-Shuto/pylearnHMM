import numpy as np
from hmmlearn import hmm

model = hmm.GaussianHMM(n_components=3, covariance_type="diag", init_params="cm", params="cmt")
model.startprob_ = np.array([1.0, 0.0, 0.0])
model.transmat_ = np.array([[0.5, 0.5, 0.0],
                            [0.0, 0.5, 0.5],
                            [0.0, 0.0, 1.0]])
model.n_features = 17

X, Z = model.sample(10)

print(X)

print(Z)
