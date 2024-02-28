#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:45:04 2019

@author: pfr
"""
#

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, islice

N=60
noises = [0., 0.1, 0.2 ]

data_sets = [datasets.make_moons(n_samples=N, noise=noises[0]),
             datasets.make_moons(n_samples=N, noise=noises[1]),
             datasets.make_moons(n_samples=N, noise=noises[2])]

names = ['one', 'two', 'three']


fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, data, noise in zip(axes.ravel(), data_sets, noises):
    
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(data[1]) + 1))))
    
    ax.set_title('std='+str(noise))
    ax.scatter(data[0][:, 0], data[0][:, 1], s=10, color=colors[data[1]])
    ax.grid()
    ax.set_xlim([-1.5, 2.5])
    ax.set_ylim([-1.5, 2.])


#fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
plt.show()

