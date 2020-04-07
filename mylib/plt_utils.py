## -------------------------------- images visualization --------------------------------- ##
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def plot_multi_voxels(*multi_voxels):
    multi_voxels = [np.array(voxels.cpu()) if isinstance(voxels, torch.Tensor) else np.array(voxels) for voxels in multi_voxels]
    multi_voxels = [np.expand_dims(voxels, 0) if voxels.ndim==3 else voxels for voxels in multi_voxels]

    rows = len(multi_voxels[0])
    columns = len(multi_voxels)
    fig = plt.figure(figsize=[10*columns,8*rows])
    for row in range(rows):
        for column in range(columns):
            if row<len(multi_voxels[column]):
                ax = fig.add_subplot(rows,columns,row*columns+column+1, projection='3d')
                ax.voxels(multi_voxels[column][row], edgecolor='k')

def plot_multi_shapes(*multi_shapes):
    multi_shapes = [np.array(shapes.cpu()) if isinstance(shapes, torch.Tensor) else np.array(shapes) for shapes in multi_shapes]
    multi_shapes = [np.expand_dims(shapes, 0) if shapes.ndim==2 else shapes for shapes in multi_shapes]

    rows = len(multi_shapes[0])
    columns = len(multi_shapes)
    fig = plt.figure(figsize=[10*columns,8*rows])
    for row in range(rows):
        for column in range(columns):
            if row<len(multi_shapes[column]):
                ax = fig.add_subplot(rows,columns,row*columns+column+1)
                ax.imshow(multi_shapes[column][row])

## -------------------------------- end of images visualization --------------------------------- ##



