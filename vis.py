from numpy import genfromtxt
from pyquaternion import Quaternion
import IPython
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from open3d import *
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

gt = genfromtxt(sys.argv[1], delimiter=' ')
data = genfromtxt(sys.argv[2], delimiter=' ')

initRotarget = numpy.linalg.inv(Quaternion( gt[0][7], gt[0][4], gt[0][5], gt[0][6]).rotation_matrix)
initPosGt = numpy.array([gt[0][1], gt[0][2], gt[0][3]]).reshape([1,3])

initRotData = numpy.linalg.inv(Quaternion( data[0][7], data[0][4], data[0][5], data[0][6]).rotation_matrix)
initPosData = numpy.array([data[0][1], data[0][2], data[0][3]]).reshape([1,3])

gtPosMod = gt[:,1:4] - initPosGt
dataPosMod = (data[:,1:4] - initPosData)


gtPosMod = gtPosMod.transpose()
for i in range(gtPosMod.shape[1]):
	gtPosMod[:,i] = numpy.matmul(initRotarget,gtPosMod[:,i])
gtPosMod = gtPosMod.transpose()

dataPosMod = dataPosMod.transpose()
for i in range(dataPosMod.shape[1]):
	dataPosMod[:,i] = numpy.matmul(initRotData,dataPosMod[:,i])
dataPosMod = dataPosMod.transpose()

source = PointCloud()
target = PointCloud()

target.points = Vector3dVector(gtPosMod)
source.points = Vector3dVector(dataPosMod)

fig =  plt.figure()
ax = plt.axes(projection='3d')

gtPosMod=numpy.asarray(target.points)
dataPosMod = numpy.asarray(source.points)


ax.plot(gtPosMod[:,0], gtPosMod[:,1], gtPosMod[:,2], 'blue', label='gt')
ax.plot(dataPosMod[:,0], dataPosMod[:,1], dataPosMod[:,2], 'red', label='data')
ax.legend()

plt.show()

IPython.embed()
