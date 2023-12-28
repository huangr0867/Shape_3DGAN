# 3D-GAN for Shape Generation

## Introduction
* Aim: Create a dataset of 3D shapes (spheres) and train a neural network to generate them.
* Generative Architecture: 3DCNN based on the paper: [https://arxiv.org/abs/1610.07584](https://arxiv.org/abs/1610.07584); done using the pytorch implementation

## Generating Dataset
The dataset was made by randomly generating point clouds of spheres. The code/procedure for the dataset generation can be found in ```shape_procedure/generate3dShapes.ipynb```. Two sample datasets with 1000 spheres are provided: point clouds with 500 points and with 1000 points (found in ```shape_data/sphere```).

## Training Model
Note: dataset and output path should be changed accordingly in ```model/constants.py```
* In the terminal, run `cd model` and then:
```python
python main.py
```
* The model will be trained for 300 epochs (specifications can be changed in ```model/constants.py```, specifics are found in ```model/trainer.py```)
* Trained model and images of generated spheres will be outputed in a separate data file (example output found in ```data/generated```)

## Pre-trained Model
* Pretrained generator and discriminators are found in ```data/generated/models```
* To use pretrained model, run
```
python main.py --pretrained=true
```
* Specific are found in ```model/generate.py```

## Results
This is an example sphere generated for a sphere with 500 points:
<img src="data/generated/pretrained_generated/pretrained_generated.png" alt="500pts_result" width="200"/>

This is an example sphere generated for a sphere with 1000 points:
<img src="results/299.png" alt="1000pts_result" width="200"/>

## Future Directions
* Adaptation for different geometric shapes and even more random shapes like biological structures
* Generation of shapes with different sizes in the dataset


