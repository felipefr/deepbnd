import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Sobol
from skopt.sampler import Lhs
from skopt.sampler import Halton
from skopt.sampler import Hammersly
from skopt.sampler import Grid
from scipy.spatial.distance import pdist

def plot_searchspace(x, title):
    fig, ax = plt.subplots()
    plt.plot(np.array(x)[:, 0], np.array(x)[:, 1], 'bo', label='samples')
    plt.plot(np.array(x)[:, 0], np.array(x)[:, 1], 'bo', markersize=80, alpha=0.5)
    # ax.legend(loc="best", numpoints=1)
    ax.set_xlabel("X1")
    ax.set_xlim([-5, 10])
    ax.set_ylabel("X2")
    ax.set_ylim([0, 15])
    plt.title(title)

n_samples = 50

space = Space([(-5., 10.), (0., 15.)])
# space.set_transformer("normalize")

np.random.seed(5)
x = np.array(space.rvs(n_samples))
plt.figure(1)
# plt.title("Random samples")
plt.scatter(x[:,0],x[:,1], label = 'random')
# pdist_data = []
# x_label = []
# pdist_data.append(pdist(x).flatten())
# x_label.append("random")

np.random.seed(7)
lhs = Lhs(lhs_type="centered", criterion=None)
x0 = np.array(lhs.generate(space.dimensions, n_samples))

lhs2 = Lhs(criterion="maximin", iterations=10000)
x2 = np.array(lhs2.generate(space.dimensions, n_samples))

sobol = Sobol()
x1 = np.array(sobol.generate(space.dimensions, n_samples))



# plt.figure(1)
# plt.title('centered LHS')
plt.scatter(x0[:,0],x[:,1], label = 'LHS')
plt.scatter(x2[:,0],x[:,1], label = 'LHS maxmin')
plt.scatter(x1[:,0],x[:,1], label = 'Sobol')
# plot_searchspace(x, )
# pdist_data.append(pdist(x).flatten())
# x_label.append("center")

plt.legend(loc = 'best')


