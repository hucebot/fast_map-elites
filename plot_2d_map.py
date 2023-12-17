#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import Voronoi, voronoi_plot_2d
import sys
from matplotlib.ticker import FuncFormatter
#from sklearn.neighbors import KDTree
import matplotlib.cm as cm
import math

my_cmap = cm.viridis

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Source: https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def load_data(filename, dim,dim_x):
    print("Loading ",filename)
    data = np.loadtxt(filename)
    fit = data[:, 0:1]
    desc = data[:,1: dim+1]
    x = data[:,dim+1:dim+1+dim_x]

    return fit, desc, x

def load_centroids(filename):
    points = np.loadtxt(filename)
    return points

def plot_cvt(ax, centroids, fit, min_fit, max_fit):
    # compute Voronoi tesselation
    print("Voronoi...")
    vor = Voronoi(centroids[:,0:2])
    regions, vertices = voronoi_finite_polygons_2d(vor)
    print("fit:", min_fit, max_fit)
    norm = matplotlib.colors.Normalize(vmin=min_fit, vmax=max_fit)
#    print("KD-Tree...")
#    kdt = KDTree(centroids, leaf_size = 30, metric = 'euclidean')

    print("plotting contour...")
    #ax.scatter(centroids[:, 0], centroids[:,1], c=fit)
    # contours
    for i, region in enumerate(regions):
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor='black', facecolor='white', lw=1)

    print("plotting data...")
    k = 0
    for i in range(0, len(centroids)):
       # q = kdt.query([desc[i]], k = 1)
       # index = q[1][0][0]
        region = regions[i]
        polygon = vertices[region]
        if fit[i] > 0:
            f = math.log(fit[i])
        else:
            f = min_fit
        if f < min_fit:
            ax.fill(*zip(*polygon), alpha=0.9, color='black')
        else:
            ax.fill(*zip(*polygon), alpha=0.9, color=my_cmap(norm(f)))
        
        k += 1
        if k % 100 == 0:
            print(k, end=" ", flush=True)
   # fit_reshaped = fit.reshape((len(fit),))
#    sc = ax.scatter(desc[:,0], desc[:,1], c=fit_reshaped, cmap=my_cmap, s=10, zorder=0)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('Usage: %s centroids_file archive.dat [min_fit] [max_fit]' % sys.argv[0])

    centroids = np.loadtxt(sys.argv[1])
    fit = np.loadtxt(sys.argv[2])    
    print("Fitness max : ", max(fit))
    index = np.argmax(fit)
    print("Average fit:", fit.sum() / fit.shape[0])

    if len(sys.argv) > 3:
        min_fit = float(sys.argv[3])
        max_fit = float(sys.argv[4])
    else:
        min_fit = min(fit)
        max_fit = math.log(max(fit))
    print("Min = {} Max={}".format(min_fit, max_fit))
    # Plot
    fig, axes = plt.subplots(1, 1, figsize=(10, 10), facecolor='white', edgecolor='white')
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    plot_cvt(axes, centroids, fit, min_fit, max_fit)
    fig.savefig('cvt.pdf')
    fig.savefig('cvt.png')
