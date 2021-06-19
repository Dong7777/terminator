
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X1=np.load('concatenate_1.npy')
X1 =X1.reshape(4704,3936)
y = np.array([0] * 2352 + [1] * 2352)

# X2=np.load('concatenate_2.npy')
# X2 =X2.reshape(4704,3936)
#
#
# X3=np.load('capsule_1.npy')
# X3 =X3.reshape(4704,574)


X4=np.load('Attention.npy')
X4 =X4.reshape(4704,896)

model = TSNE(learning_rate=200, n_components=2, random_state=0, perplexity=200)
tsne1 = model.fit_transform(X1)

# model = TSNE(learning_rate=200, n_components=2, random_state=0, perplexity=200)
# tsne2 = model.fit_transform(X2)
#
# model = TSNE(learning_rate=200, n_components=2, random_state=0, perplexity=200)
# tsne3 = model.fit_transform(X3)

model = TSNE(learning_rate=200, n_components=2, random_state=0, perplexity=200)
tsne4 = model.fit_transform(X4)


plt.suptitle('Rat brain', fontsize=20)
plt.subplots_adjust(left=0.2, wspace=0.9, top=0.8)

plt.figure(1)
plt.subplot(121)
scatter =plt.scatter(tsne1[:, 0], tsne1[:, 1], c=y,s=0.5, alpha = 1)
# plt.text(1, 1, "input", size=20, rotation=0.,ha="right", va="baseline",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
plt.title('Concatenate', fontsize=12)
a,b=scatter.legend_elements()
b=['$\\mathdefault {m^6A}$',
 '$\\mathdefault{non-m^6A}$']
legend1 = plt.legend(a,b, title="Classes")

# plt.text(1, 1, "input", size=20, rotation=0.,ha="right", va="baseline",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))


# plt.subplot(412)
# plt.scatter(tsne2[:, 0], tsne2[:, 1], c=y,s=0.5, alpha = 1)
# plt.title("concatenate_2")
#
# plt.subplot(413)
# plt.scatter(tsne3[:, 0], tsne3[:, 1], c=y,s=0.5, alpha = 1)
# plt.title("capsule")
plt.subplot(122)
scatter =plt.scatter(tsne4[:, 0], tsne4[:, 1], c=y,s=0.5, alpha = 1)
a,b=scatter.legend_elements()
b=['$\\mathdefault {m^6A}$',
 '$\\mathdefault{non-m^6A}$']
legend1 = plt.legend(a,b, title="Classes")

plt.title("Attention", fontsize=12)





plt.show()



