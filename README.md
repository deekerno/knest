<p align="center">
  <img src="https://i.imgur.com/hnx8fkF.png" alt="Vector Bird Logo">
</p>

Table of Contents
=================
* [Description](#description)
* [Organization](#organization)
* [Application Stages](#application-stages)
* [Additional Setup](#additional-setup)
* [Testing](#testing)
* [Windows Development](#windows-development)

## Description
Take a folder that contains photos of birds and other stuff, get a folder of just birds, no other stuff.

## Organization
+ architectures - neural network models
+ assets - GUI assets
+ config - hyperparameter configurations for models
+ tests - unit tests
+ utils - image utilities

## Application Stages
1. [Blur Detection](https://github.com/adcrn/knest/wiki/Blur-Detection)
2. [Object Classification](https://github.com/adcrn/knest/wiki/Object-Classification)
3. [Object Localization](https://github.com/adcrn/knest/wiki/Object-Localization)
4. [Image Comparison](https://github.com/adcrn/knest/wiki/Image-Comparison)

## Additional Setup
Due to GitHub's file size limit of 100MB, much of our object localization network configuration cannot be hosted directly on GitHub. To alleviate this process, we are using Git Large File Storage (GLFS) in order to provide a seamless Git experience. In the case of datasets, Google Drive was used to facilitate ease of transfers.

## Testing
All unit tests can be found in the `tests` folder. The test suite can be run by using the following command:
```
python3 -m unittest discover
```
More information about testing can be found [here](https://github.com/adcrn/knest/wiki/Testing).

## Windows Development
If developing for this application on Windows, there are a number of issues of which one should be aware. Those issues, and their solutions, can be found [here](https://github.com/adcrn/knest/wiki/Windows-Development).

