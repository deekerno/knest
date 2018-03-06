<p align="center">
  <img src="https://i.imgur.com/hnx8fkF.png" alt="Vector Bird Logo">
</p>

## Description
Take a folder that contains photos of birds and other stuff, get a folder of just birds, no other stuff.

## Structure
+ architectures - neural network models
+ assets - GUI assets
+ config - hyperparameter configurations for models
+ tests - unit tests
+ utils - image utilities

## Testing
All unit tests can be found in the `tests` folder. The test suite can be run by using the following command:
```
python3 -m unittest discover
```

### Training Dataset
To ensure accuracy of the training model, we gathered a training dataset of large, high-resolution images. The set is divided into two classifications: bird and not_bird. File names of images that contain birds will begin with `00`, and those without birds will begin with `10`.

