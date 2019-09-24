import numpy as np
from scipy import ndimage
from skimage import measure, morphology

def find_edges(mask, level=0.5):
    edges = measure.find_contours(mask, level)[0]
    print(type(edges))
    ys = edges[:, 0]
    xs = edges[:, 1]
    return xs, ys

def plot_contours(arr, aux=None, level=0.5, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)
    ax.imshow(arr, cmap=plt.cm.gray)
    if aux is not None:
        xs, ys = find_edges(aux, level)
        ax.plot(xs, ys)
        
def preprocess_seg(seg, spacing, new_spacing=[1., 1., 1.], smooth=1):
    '''a standard preprocessing for voxel and (nodule) segmentation.
    resize to 1mm x 1mm x 1mm (or `new_spacing`),
    then convert HU into [-1024,400],
    smoothing will work on the segmentation if not None.
    '''
    resized_seg, _ = resize(seg, spacing, new_spacing)
    if smooth is not None:
        smoothed_seg = morphology.binary_closing(resized_seg, morphology.ball(smooth))
    else:
        smoothed_seg = resized_seg.copy()
    return smoothed_seg
def preprocess_voxel(voxel,spacing,
                     window_low=-1024, window_high=400,
                     new_spacing=[1., 1., 1.], cast_dtype=np.uint8, smooth=1):
    resized_voxel, _ = resize(voxel, spacing, new_spacing)
    mapped_voxel = window_clip(resized_voxel, window_low, window_high, dtype=cast_dtype)
    return mapped_voxel

def crop_at_zyx_with_dhw(voxel, zyx, dhw, fill_with):
    '''Crop and pad on the fly.'''
    shape = voxel.shape
    # z, y, x = zyx
    # d, h, w = dhw
    crop_pos = []
    padding = [[0, 0], [0, 0], [0, 0]]
    for i, (center, length) in enumerate(zip(zyx, dhw)):
        assert length % 2 == 0
        # assert center < shape[i] # it's not necessary for "moved center"
        low = round(center) - length // 2
        high = round(center) + length // 2
        if low < 0:
            padding[i][0] = int(0 - low)
            low = 0
        if high > shape[i]:
            padding[i][1] = int(high - shape[i])
            high = shape[i]
        crop_pos.append([int(low), int(high)])
    cropped = voxel[crop_pos[0][0]:crop_pos[0][1], crop_pos[1][0]:crop_pos[1][1], crop_pos[2][0]:crop_pos[2][1]]
    if np.sum(padding) > 0:
        cropped = np.lib.pad(cropped, padding, 'constant',
                             constant_values=fill_with)
    return cropped

def window_clip(v, window_low=-1024, window_high=400, dtype=np.uint8):
    '''Use lung windown to map CT voxel to grey.'''
    # assert v.min() <= window_low
    return np.round(np.clip((v - window_low) / (window_high - window_low) * 255., 0, 255)).astype(dtype)




def resize(voxel, spacing, new_spacing=[1., 1., 1.]):
    '''Resize `voxel` from `spacing` to `new_spacing`.'''
    resize_factor = []
    for sp, nsp in zip(spacing, new_spacing):
        resize_factor.append(float(sp) / nsp)
    resized = ndimage.interpolation.zoom(
        voxel, resize_factor, mode='nearest')
    for i, (sp, shape, rshape) in enumerate(zip(spacing, voxel.shape, resized.shape)):
        new_spacing[i] = float(sp) * shape / rshape
    return resized, new_spacing

def rotation(array, angle):
    '''using Euler angles method.
    @author: renchao
    @params:
        angle: 0: no rotation, 1: rotate 90 deg, 2: rotate 180 deg, 3: rotate 270 deg
    '''
    #
    X = np.rot90(array, angle[0], axes=(0, 1))  # rotate in X-axis
    Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
    Z = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return Z


def reflection(array, axis):
    '''
    @author: renchao
    @params:
        axis: -1: no flip, 0: Z-axis, 1: Y-axis, 2: X-axis
    '''
    if axis != -1:
        ref = np.flip(array, axis)
    else:
        ref = np.copy(array)
    return ref


def crop(array, zyx, dhw):
    z, y, x = zyx
    d, h, w = dhw
    cropped = array[z - d // 2:z + d // 2,
              y - h // 2:y + h // 2,
              x - w // 2:x + w // 2]
    return cropped

def random_center(shape, move):
    offset = np.random.randint(-move, move + 1, size=3)
    zyx = np.array(shape) // 2 + offset
    return zyx
