# 3D-GAN for Shape Generation

## Introduction
* Aim: Create a dataset of 3D shapes (spheres) and train a neural network to generate them.
* Generative Architecture: 3DCNN based on the paper: [https://arxiv.org/abs/1610.07584](https://arxiv.org/abs/1610.07584); done using the pytorch implementation
* More info outlined in write_up.pdf

## Generating Dataset
The dataset was made by randomly generating point clouds of spheres. The code/procedure for the dataset generation can be found in ```shape_procedure/generate3dShapes.ipynb```. Two sample datasets with 1000 spheres are provided: point clouds with 500 points and with 1000 points (found in ```shape_data/sphere```).

For a filled in set of spheres, it can be found in the `filled` folder in `shape_data`.

## Training Model
Note: dataset and output path should be changed accordingly in ```model/constants.py```
* In the terminal, run `cd model` and then:
```python
python main.py
```
* The model will be trained for 1000 epochs and a batch size of 2. To change the specifications, run:
```python
python main.py --epochs=[EPOCH] --batch_size=[BATCH_SIZE]
```
with replaced values for [EPOCH] and [BATCH_SIZE].
* Trained model and images of generated spheres will be outputed in a separate data file (example output found in ```data/generated```).

## Pre-trained Model
* Pretrained generator and discriminators are found in ```data/generated/models```
* To use pretrained model, run `cd model` and then:
```
python main.py --pretrained=true
```
* Specifics are found in ```model/generate.py```

## Results
These are some example generated spheres:

<img src="data/generated/pretrained_generated/pretrained_generated0.png.png" alt="1000pts_result" width="500"/>

<img src="data/generated/pretrained_generated/pretrained_generated8.png.png" alt="1000pts_result" width="500"/>

## Future Directions
* Adaptation for different geometric shapes and even more random shapes like biological structures
* Generation of shapes with different sizes in the dataset

## References
* Pytorch implemenation based on the repo: [3DGAN-Pytorch](https://github.com/rimchang/3DGAN-Pytorch/tree/master)
* Pytorch usage reference: [DCGAN Pytorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
* Original paper: [https://arxiv.org/abs/1610.07584](https://arxiv.org/abs/1610.07584)


