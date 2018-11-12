import numpy as np

def image_normalize(image, mask, map_min, map_max):

    '''
    This normalization is linearly mapping the voxel value in the mask to the value map_min ~ map_max.
    
    # Inputs:
        image: Original image.
        mask: Original mask.
        map_min: The minimum value that you want to map.
        map_max: The maximum value that you want to map.
        
    # Output:
        changed_image: The normalized image.
    '''
    
    if mask.any() > 0:
    
        changed_image=image

        M=np.max(image[mask>0])
        m=np.min(image[mask>0]) 

        changed_image[mask>0]=(changed_image[mask>0]-m)*(map_max - map_min)/(M-m)
    
    else:
        changed_image = image
    
    return changed_image


def combine_image_mask(image, mask):
    
    '''
    This function is combining the image and the mask.
    
    # Inputs:
        image: Original image.
        mask: Original mask.
        
    # Output:
        combine_image_mask: Combine image and mask.
    '''
    
    value = np.max(image)*1.1;
    
    combine_image_mask = np.copy(image)
    combine_image_mask[mask == 1] = value
    
    return combine_image_mask
    