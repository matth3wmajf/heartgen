# HeartGenerator
## Introduction
HeartGenerator, or `heartgen`, is a small program written in C that uses a feedforward neural network as an autoencoder to generate images of hearts, made in dedication to my girlfriend, who's been nothing but an amazing person to me.
Although this program is made for generating images of hearts, it can easily be modified to generate all kinds of other images.
The dataset here is made by me, and contains images of hearts, in the form of 128x128 bitmap-formatted images. This dataset is very small, and I am not done adding to it.
This program uses the [Alpha](https://github.com/matth3wmajf/alpha) library, which is a lightweight library that provides useful AI-related constructs & mathematical functions.
## Usage
Currently, I've got two specific models, one consisting of a feedforward neural network that consists of one hidden layer that is 512 neurons wide (`heartgen-a1.bin`), and another consisting of three hidden layers, with the first one being 256 neurons wide, the second being 512 neurons wide, and the third being 256 neurons wide (`heartgen-b1.bin`). I may create more of these different models in the future, to test which ones perform the best, which ones produce the most interesting results, and so on.
For now, I am not keeping these models in the repository, as `git` doesn't like large files, and I don't want to take up too much space on GitHub.
To use these models, you can invoke the `heartgen` command, specifying the model that you wish to use, and the file name of the image that you wish to generate:
```
heartgen --generate path/to/model.bin path/to/output.bmp
```
For example, let's say you want to generate an image of a heart using the `heartgen-a1.bin` model, and you want to save the image to `heartgen-000.bmp`. You would invoke the command like so:
```
heartgen --generate heartgen-a1.bin heartgen-000.bmp
```
You can also train the model yourself, by invoking the `heartgen` command with the `--train` flag, specifying the model that you wish to train, and the image's file name:
```
heartgen --train path/to/model.bin path/to/image.bmp
```
If the model doesn't exist, it'll be created.