# 3D-GAN for Shape Generation

## Introduction
* Aim: Create a dataset of 3D shapes (spheres) and train a neural network to generate them.
* Generative Architecture: 3DCNN based on the paper [https://arxiv.org/abs/1610.07584](https://arxiv.org/abs/1610.07584)

### Generating Dataset
The dataset was made by randomly generating point clouds of spheres. The code/procedure for the dataset generation can be found in ```generate_shapes/generate3dShapes.ipynb```. Two sample datasets with 1000 spheres are provided: point clouds with 500 points and with 1000 points (found in ```shape_data/sphere```).

### Training Model
* In the terminal, run `cd model` and then `python main.py`
* The model will be trained for 500 epochs (specifications can be changed in ```model/constants.py```)
* Trained model and images of generated spheres will be outputed in a separate data file

### Results

