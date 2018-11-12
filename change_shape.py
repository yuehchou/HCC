import numpy as np
import scipy.ndimage as ndi
import random

def transform_matrix_offset_center(matrix, x, y, o_x=None, o_y=None):
    
    if o_x == None:
        o_x = float(x) / 2 + 0.5
    if o_y == None:
        o_y = float(y) / 2 + 0.5
        
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_affine_transform(x, theta=0,
                           row_axis=0, col_axis=1, channel_axis=2,
                           o_h=None, o_w=None,
                           fill_mode='nearest', cval=0.):
    
    """Applies an affine transformation specified by the parameters given.
    
    # Inputs:
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        o_h: The x-axis center of this matrix.
        o_w: The y-axis center of this matrix.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
            
    # Output:
        The transformed version of the input.
    """
    
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if transform_matrix is not None:
        h, w = x.shape[0], x.shape[1]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w, o_h, o_w)
        x = np.rollaxis(x, 2, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndi.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, 2 + 1)
        
    return x

def rotation_2D(x, theta, o_h=None, o_w=None, fill_mode='nearest', cval=0.):
    
    """Performs a random rotation of a Numpy image tensor.
    
    # Inputs:
        x: Input tensor. Must be 3D.
        theta: Rotation degrees.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
            
    # Output:
        Rotated Numpy image tensor.
    """
    
    x = apply_affine_transform(x, theta=theta,
                               o_h=o_h, o_w=o_w,
                               fill_mode=fill_mode, cval=cval)
    return x


def rotation_3D(x, theta_x, theta_y, theta_z, cval=0.):
    
    range_mask=np.where(x > 0)
    
    if range_mask[0].shape[0] != 0 and range_mask[1].shape[0] != 0 and range_mask[2].shape[0] != 0:
        o_x = (np.max(range_mask[0]) + np.min(range_mask[0])) / 2 + 0.5
        o_y = (np.max(range_mask[1]) + np.min(range_mask[1])) / 2 + 0.5
        o_z = (np.max(range_mask[2]) + np.min(range_mask[2])) / 2 + 0.5
    else:
        o_x = None
        o_y = None
        o_z = None
    
    if theta_x != 0:
        for i in np.arange(x.shape[2]):
            x[:,:,i,:] = rotation_2D(x[:,:,i,:], theta_x, o_x, o_y, fill_mode='nearest', cval=cval)
        
    if theta_y != 0:
        x = np.transpose(x, (1, 2, 0, 3))
        for i in np.arange(x.shape[2]):
            x[:,:,i,:] = rotation_2D(x[:,:,i,:], theta_y, o_y, o_z, fill_mode='nearest', cval=cval)
        x = np.transpose(x, (2, 0, 1, 3))
    
    if theta_z != 0:
        x = np.transpose(x, (2, 0, 1, 3))
        for i in np.arange(x.shape[2]):
            x[:,:,i,:] = rotation_2D(x[:,:,i,:], theta_z, o_z, o_x, fill_mode='nearest', cval=cval)
        x = np.transpose(x, (1, 2, 0, 3))
    
    return x


def translation_3D(img, trans_x, trans_y, trans_z, cval=0.):

    """Performs a translation of a Numpy image tensor.
    
    # Inputs:
        img: Input tensor. Must be 3D.
        trans_x: The distance that you want to move in the x-axis.
        trans_y: The distance that you want to move in the y-axis.
        trans_z: The distance that you want to move in the z-axis.
        cval: Value used for points outside the boundaries of the input.
        
    # Output:
        Translation Numpy image tensor.
    """
    
    if trans_x > 0:
        img[trans_x:,...] = img[:-trans_x,...] 
        img[:trans_x,...] = cval
    elif trans_x < 0:
        img[:trans_x,...] = img[-trans_x:,...] 
        img[trans_x:,...] = cval
    
    if trans_y > 0:
        img[:,trans_y:,:,:] = img[:,:-trans_y,:,:] 
        img[:,:trans_y,:,:] = cval
    elif trans_y < 0:
        img[:,:trans_y,:,:] = img[:,-trans_y:,:,:] 
        img[:,trans_y:,:,:] = cval
        
    if trans_z > 0:
        img[...,trans_z:,:] = img[...,:-trans_z,:] 
        img[...,:trans_z,:] = cval
    elif trans_z < 0:
        img[...,:trans_z,:] = img[...,-trans_z:,:] 
        img[...,trans_z:,:,:] = cval
    
    return img


def zoom_3D(img, x_zoom_range, y_zoom_range, z_zoom_range, cval=0.):
    
    """Performs a zoom of a Numpy image tensor.
    
    # Inputs:
        img: Input tensor. Must be 3D.
        x_zoom_range: Should be greater than 0.
        y_zoom_range: Should be greater than 0.
        z_zoom_range: Should be greater than 0.
        
        If the range is between 0 and 1, the image will be zoom-out（放大）.
        If the range is greater than 1, the image will be zoom-in.
        
    # Output:
        Zoom Numpy image tensor if x_zoom_range, y_zoom_range and z_zoom_range are greater than 0,
        if NOT, it will return oringinal image.
        
    """
    
    h, w, t = img.shape[0], img.shape[1], img.shape[2]

    o_h = float(h) / 2 - 0.5
    o_w = float(w) / 2 - 0.5
    o_t = float(t) / 2 - 0.5
    
    if x_zoom_range > 0 and y_zoom_range > 0 and z_zoom_range > 0:
    
        if x_zoom_range != 1 and x_zoom_range > 0:
            image=np.empty(img.shape) 
            
            origin_idx = np.arange(h)
    
            zoom_idx = np.full(h, o_h) + (origin_idx - o_h) * x_zoom_range
            zoom_idx[zoom_idx < 0] = -2
            zoom_idx[zoom_idx > (h-1)] = -2

            floor_zoom_idx = zoom_idx.astype(int)
            celling_zoom_idx = (zoom_idx + 1).astype(int)

            for i in range(h):
                if zoom_idx[i] < 0:
                    image[i,:,:,:] = cval
                else:
                    image[i,:,:,:] = img[floor_zoom_idx[i],:,:,:]*(celling_zoom_idx[i] - zoom_idx[i]) + img[celling_zoom_idx[i],:,:,:]*(zoom_idx[i] - floor_zoom_idx[i])
            
            img=np.copy(image)

        if y_zoom_range != 1 and Y_zoom_range > 0:
            image=np.empty(img.shape)
            
            origin_idx = np.arange(w)
    
            zoom_idx = np.full(origin_idx.shape, o_w) + (origin_idx - o_w) * y_zoom_range
            zoom_idx[zoom_idx < 0] = -2
            zoom_idx[zoom_idx > (w-1)] = -2

            floor_zoom_idx = zoom_idx.astype(int)
            celling_zoom_idx = (zoom_idx + 1).astype(int)
            
            for i in range(w):
                if zoom_idx[i] < 0:
                    img[:,i,:,:] = cval
                else:
                    img[:,i,:,:] = img[:,floor_zoom_idx[i],:,:]*(celling_zoom_idx[i] - zoom_idx[i]) + img[:,celling_zoom_idx[i],:,:]*(zoom_idx[i] - floor_zoom_idx[i])
                    
            img=np.copy(image)
            
            
            
        if z_zoom_range != 1 and  z_zoom_range > 0:
            image=np.empty(img.shape)
            
            origin_idx = np.arange(t)
    
            zoom_idx = np.full(origin_idx.shape, o_t) + (origin_idx - o_t) * z_zoom_range
            zoom_idx[zoom_idx < 0] = -2
            zoom_idx[zoom_idx > (t-1)] = -2

            floor_zoom_idx = zoom_idx.astype(int)
            celling_zoom_idx = (zoom_idx + 1).astype(int)
            
            for i in range(t):
                if zoom_idx[i] < 0:
                    image[:,:,i,:] = cval
                else:
                    image[:,:,i,:] = img[:,:,floor_zoom_idx[i],:]*(celling_zoom_idx[i] - zoom_idx[i]) + img[:,:,celling_zoom_idx[i],:]*(zoom_idx[i] - floor_zoom_idx[i])
                    
            img=np.copy(image)
            

    return img


def mask_zoom_3D(img, x_zoom_range, y_zoom_range, z_zoom_range, cval=0.):
    
    """Performs a zoom of a Numpy image tensor.
    
    # Inputs:
        img: Input tensor. Must be 3D.
        x_zoom_range: Should be greater than 0.
        y_zoom_range: Should be greater than 0.
        z_zoom_range: Should be greater than 0.
        cval: The value that you want to fill the empty area. 
        
        If the range is between 0 and 1, the image will be zoom-out（放大）.
        If the range is greater than 1, the image will be zoom-in (縮小).
        
    # Output:
        Zoom Numpy image tensor if x_zoom_range, y_zoom_range and z_zoom_range are greater than 0,
        if NOT, it will return oringinal image.
    """
    
    img = img.astype(float)
    
    
    range_mask=np.where(img > 0)
    
    h, w, t = img.shape[0], img.shape[1], img.shape[2]
    
    if range_mask[0].shape[0] != 0 and range_mask[1].shape[0] != 0 and range_mask[2].shape[0] != 0:
        o_h = (np.max(range_mask[0]) + np.min(range_mask[0])) / 2 + 0.5
        o_w = (np.max(range_mask[1]) + np.min(range_mask[1])) / 2 + 0.5
        o_t = (np.max(range_mask[2]) + np.min(range_mask[2])) / 2 + 0.5
    else:
        o_h = float(h) / 2 - 0.5
        o_w = float(w) / 2 - 0.5
        o_t = float(t) / 2 - 0.5
    
    if x_zoom_range > 0 and y_zoom_range > 0 and z_zoom_range > 0:
    
        if x_zoom_range != 1 and x_zoom_range > 0:
            image=np.empty(img.shape, dtype=float) 
            
            origin_idx = np.arange(h)
    
            zoom_idx = np.full(h, o_h) + (origin_idx - o_h) * x_zoom_range
            zoom_idx[zoom_idx < 0] = -2
            zoom_idx[zoom_idx > (h-1)] = -2

            floor_zoom_idx = zoom_idx.astype(int)
            floor_zoom_idx = floor_zoom_idx.astype(float)
            celling_zoom_idx = (zoom_idx + 1).astype(int)
            celling_zoom_idx = celling_zoom_idx.astype(float)

            for i in range(h):
                if zoom_idx[i] < 0:
                    image[i,:,:,:] = cval
                else:
                    image[i,:,:,:] = img[int(floor_zoom_idx[i]),:,:,:]*float(celling_zoom_idx[i] - zoom_idx[i]) + img[int(celling_zoom_idx[i]),:,:,:]*float(zoom_idx[i] - floor_zoom_idx[i])
    
            image[image>0.5] = 1
            image[image<0.5] = 0
            img=np.copy(image)

        if y_zoom_range != 1 and y_zoom_range > 0:
            image=np.empty(img.shape, dtype=float)
            
            origin_idx = np.arange(w)
    
            zoom_idx = np.full(origin_idx.shape, o_w) + (origin_idx - o_w) * y_zoom_range
            zoom_idx[zoom_idx < 0] = -2
            zoom_idx[zoom_idx > (w-1)] = -2

            floor_zoom_idx = zoom_idx.astype(int)
            floor_zoom_idx = floor_zoom_idx.astype(float)
            celling_zoom_idx = (zoom_idx + 1).astype(int)
            celling_zoom_idx = celling_zoom_idx.astype(float)
            
            for i in range(w):
                if zoom_idx[i] < 0:
                    image[:,i,:,:] = cval
                else:
                    image[:,i,:,:] = img[:,int(floor_zoom_idx[i]),:,:]*float(celling_zoom_idx[i] - zoom_idx[i]) + img[:,int(celling_zoom_idx[i]),:,:]*float(zoom_idx[i] - floor_zoom_idx[i])
    
            image[image>0.5] = 1
            image[image<0.5] = 0
            img=np.copy(image)
            
        if z_zoom_range != 1 and z_zoom_range > 0:
            image=np.empty(img.shape, dtype=float)
            
            origin_idx = np.arange(t)
    
            zoom_idx = np.full(origin_idx.shape, o_t) + (origin_idx - o_t) * z_zoom_range
            zoom_idx[zoom_idx < 0] = -2
            zoom_idx[zoom_idx > (t-1)] = -2

            floor_zoom_idx = zoom_idx.astype(int)
            floor_zoom_idx = floor_zoom_idx.astype(float)
            celling_zoom_idx = (zoom_idx + 1).astype(int)
            celling_zoom_idx = celling_zoom_idx.astype(float)
            
            for i in range(t):
                if zoom_idx[i] < 0:
                    image[:,:,i,:] = cval
                else:
                    image[:,:,i,:] = img[:,:,int(floor_zoom_idx[i]),:]*float(celling_zoom_idx[i] - zoom_idx[i]) + img[:,:,int(celling_zoom_idx[i]),:]*float(zoom_idx[i] - floor_zoom_idx[i])
                    
            image[image>0.5] = 1
            image[image<0.5] = 0
            img=np.copy(image)
            

    return img



def Random_change_shape(mask, rx, ry, rz, trans_x, trans_y, trans_z,
                        x_zoom_range, y_zoom_range, z_zoom_range, cval):
    
    """No channel"""
    
    size=mask.shape
    dimension=len(size)
    
    changed_mask=np.zeros([size[0],size[1],size[2],1])
    
    changed_mask[...,0]=mask
    
    # random rotation 
    if random.choice([True,False]):
        changed_mask=rotation_3D(changed_mask, rx, ry, rz, cval)
        
    # random translation
    if random.choice([True,False]):
        changed_mask=translation_3D(changed_mask, trans_x, trans_y, trans_z, cval)
        
    # random zoom
    if random.choice([True,False]):
        changed_mask=mask_zoom_3D(changed_mask, x_zoom_range, y_zoom_range, z_zoom_range, cval)
        
    changed_mask=np.squeeze(changed_mask,3)
    
    return changed_mask
