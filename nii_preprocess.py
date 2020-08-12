import numpy as np
import SimpleITK as sitk
import multiprocessing


def niftiToFlowArray(file_list, image_height, image_width):
    """
    Process a list of Nifti files and return two Rank 4 Numpy arrays
    for using the Keras ImageDataGenerator.flow() method
    """
    image_batch = []
    for file_zip in file_list:
        nifti_num = int(file_zip[0].replace('.nii','')[-1])
        image_batch.append(parallelSlices(file_zip, nifti_num, image_height, image_width))
    
    image_array, mask_array = createFlowArray(image_batch, image_height, image_width)
    
    return image_array, mask_array


def findSlicesWithMasks(ind, nifti_num, img, mask, image_batch, image_height, image_width):
    # This returns images which have corresponding masks
    mask_slice = mask[:,:,ind]
    if max(mask_slice) > 0:
        img_slice = img[:,:,ind]
        img_slice = (img_slice - min(img_slice))/(max(img_slice) - min(img_slice))
        img_slice = sitk.GetArrayFromImage(img_slice)
        img_slice = np.reshape(img_slice, (1, image_height, image_width, 1))
        img_slice = img_slice.astype(dtype='float32')
        mask_slice = sitk.GetArrayFromImage(mask_slice)
        mask_slice = np.reshape(mask_slice, (1, image_height, image_width, 1))
        mask_slice = mask_slice.astype(dtype='float32')
        image_batch[ind] = (nifti_num, ind, img_slice, mask_slice)
        
    return image_batch
        
    
def createFlowArray(image_batch, image_height, image_width, num_channels=1):
    """
    Batch indices:
    0: Nifti file number
    1: Slice number
    2: Image array
    3: Mask array

    Create a Rank 4 numpy array each for images and masks
    (batch size, height, width, channels=1)
    """
    image_array = np.empty(shape=(1, image_height, image_width, num_channels), dtype='float32')
    mask_array = np.empty(shape=(1, image_height, image_width, num_channels), dtype='float32')
    for nifti in image_batch:
        for entry in nifti.values():
            image_array = np.append(image_array, entry[2], axis=0)
            mask_array = np.append(mask_array, entry[3], axis=0)
            
    image_array = np.delete(image_array, 0, 0)
    mask_array = np.delete(mask_array, 0, 0)
                
    return image_array, mask_array
    
    

def parallelSlices(file_zip, nifti_num, image_height, image_width):
    """
    Iterating through the slices of each Nifti file one at a time is very slow.
    This processes the slices in parallel and returns a list of dictionaries,
    where each dictionary contains the results from a single Nifti file.
    """
    manager = multiprocessing.Manager()
    image_batch = manager.dict()
    jobs = []
    
    image = sitk.ReadImage(file_zip[0])
    mask = sitk.ReadImage(file_zip[1])
    for i in range(image.GetSize()[2]):
        p = multiprocessing.Process(target=findSlicesWithMasks, args=(i, nifti_num, image, mask, image_batch, image_height, image_width))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
        
    return image_batch





