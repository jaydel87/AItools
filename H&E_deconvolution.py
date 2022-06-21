from skimage.util import img_as_float
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.axes as axes
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, OPTICS
from matplotlib  import cm
from scipy.interpolate import interp2d
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.optimize import curve_fit

# img_as_float returns the image as an array of floats between 0 and 1
image = img_as_float(io.imread("C:/Users/jayde/IONOS HiDrive/R&D/PROJETS/BFTE_SPARSE/Datasets/WSSS4LUAD/WSSS4LUAD/2.validation/2.validation/img/34.png"))

chRed = image[:, :, 0]
chGreen = image[:, :, 1]
chBlue = image[:, :, 2]

rows, cols, channels = image.shape
print(rows, cols)
maxIntensity = 255.

ODRed = -np.log10(chRed).flatten()
ODGreen = -np.log10(chGreen).flatten()
ODBlue = -np.log10(chBlue).flatten()

ODImage = np.array([ODRed, ODBlue, ODGreen])
ODcoords = np.transpose(ODImage)

#ODcoords = ODcoords[0:5000, :]


# x = ODRed
# y = ODGreen
# z = ODBlue

x = ODcoords[:, 0]
y = ODcoords[:, 1]
z = ODcoords[:, 2]

xx = np.arange(0, round(max(ODRed), 1) + 0.1, 0.1)
yy = np.arange(0, round(max(ODGreen), 1) + 0.1, 0.1)
zz = np.arange(0, round(max(ODBlue), 1) + 0.1, 0.1)

nobins = 25
xbins = np.linspace(0, round(max(ODcoords[:, 0]), 1), nobins+1)
ybins = np.linspace(0, round(max(ODcoords[:, 1]), 1), nobins+1)
zbins = np.linspace(0, round(max(ODcoords[:, 2]), 1), nobins+1)
H, edges = np.histogramdd(ODcoords, bins=[xbins, ybins, zbins])

xcentres = []
ycentres = []

bb = np.linspace(0, nobins-1, nobins)

for i in range(0, nobins):
    xcentres.append((xbins[i+1] + xbins[i]) / 2)
    ycentres.append((ybins[i+1] + ybins[i]) / 2)

#model = KMeans(n_clusters=2, init='k-means++', n_init=100)
# clusters = model.fit_predict(ODcoords)
# print(clusters)

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
# ax.scatter(x, y, z, color='k')
#
# xflat = np.full_like(x, min(ax.get_xlim()))
# yflat = np.full_like(y, max(ax.get_ylim()))
# zflat = np.full_like(z, min(ax.get_zlim()))
#
# ax.scatter(xflat, y, z, color='r')
# ax.scatter(x, yflat, z, color='g')
# ax.scatter(x, y, zflat, color='b')

#fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.scatter(x,y,s=20,c=z, marker = 'o', cmap = cm.jet )
# ax2.scatter(x,z,s=20,c=y, marker = 'o', cmap = cm.jet )
# ax3.scatter(y,z,s=20,c=x, marker = 'o', cmap = cm.jet )

# ax1.scatter(x[clusters == 0],y[clusters == 0],s=20, marker = 'o', color='b')
# ax1.scatter(x[clusters == 1],y[clusters == 1],s=20, marker = 'o', color='r')
# ax2.scatter(x[clusters == 0],z[clusters == 0],s=20, marker = 'o', color='b')
# ax2.scatter(x[clusters == 1],z[clusters == 1],s=20, marker = 'o', color='r')
# ax3.scatter(y[clusters == 0],z[clusters == 0],s=20, marker = 'o', color='b')
# ax3.scatter(y[clusters == 1],z[clusters == 1],s=20, marker = 'o', color='r')

print(H.shape)

def gauss(x,mu,sigma,A):
    return A*np.exp((-(x-mu)**2)/(2*sigma**2))

def double_gauss(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

for i in range(0, nobins):
    array = H[:, :, i]
    xsum = []
    ysum = []
    for j in range(0, nobins):
        xsum.append(np.sum(array[j, :]))
        ysum.append(np.sum(array[:, j]))

    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(7, 7), constrained_layout=True)

    ax[0, 0].semilogy(bb, ysum)
    ax[0, 0].set_ylim([0.1, max(ysum)])

    if any(ysum) != False:
        try:
            guess = (bb[ysum.index(max(ysum))], 2, max(ysum), bb[ysum.index(max(ysum))] + (i/nobins)*10, 2, max(ysum)/10)
            params, cov = curve_fit(double_gauss, bb, ysum, guess, maxfev=5000)
            gfit = double_gauss(bb, *params)
            ax[0, 0].semilogy(bb, gfit)
        except:
            continue

    ax[1, 0].imshow(array, norm=colors.LogNorm(1e-6, H.max()), cmap=cm.jet, extent=[0, nobins, nobins, 0])
    ax[1, 1].semilogx(xsum, bb)
    ax[1, 1].set_xlim([0.1, max(xsum)])
    ax[0, 1].remove()
    #plt.show()
    plt.savefig("hist_"+str(i)+".png")


