# BDCLearn - Dockerized Keras deep learning tools for BDCat

## Jupyter Notebooks
The included Jupyter notebooks are self-contained tools for training each network type. Each notebook contains preprocessing and training with an example dataset, but the intent is that the user can replace to preprocessing section with a custom script and data can be imported.

## Pretrained models
Models pretrained with example datasets are available
<br />
UNet - trained with lung segmentation masks in the COVID-19 dataset from http://medicalsegmentation.com/covid19/
<br />
LeNet - trained with the Fashion MNist set included in Keras

## Docker


## Networks
Current : UNet\
To Do: VGG-16, LeNet, Resnet, SegNet, ...

## Commands
/keras/{network name} **new** {kwargs} : create a new hdf5 model file with the selected network structure\
/keras/{network name} **train --model_path --image_path --class_path** {kwargs} : train a model from image directories (see https://keras.io/api/preprocessing/image/#flowfromdirectory-method)
<br />
<br />
In progress:\
/keras/{network name} **test --model_path --image_path --class_path** {kwargs} : test a trained model for image directories\
/keras/{network name} **predict --model_path --image_path** : use a model to predict classes from an image directory
<br />

## Using custom preprocessing scripts in Python
Most use cases will involve customized preprocessing or directory trees not compatible with the Keras flow_from_directory method. By using the **new** command (see above), the model can be loaded into a custom script without having to instantiate it manually. The scripts in the **util** directory contain functions that can be called from a custom script to simplify coding for training, testing, and prediction.\
**This section will be expanded with a detailed description of the available functions**
