# Project: Search and Sample Return

**Sergei Surovtsev**
<br/>
Udacity Robotics Software Engineer Nanodegree
<br/>
Class of November 2018

## Project Description

This project is introduction to Deep Learning for computer vision. it involves processing camera feed of a quad-rotor to segment image into 3 categories: foreground, people and hero. We are solving segmentation by applying Fully Convolutional Network (FCN) to a problem. I've also explored more recent architectures and experimented with constructing new ones.

## Project Goals

* Introduction to Deep Neural Networks (DNNs)
* Introduction to vision processing with DNNs with Convolutional Networks (CNNs)
* Introduction to Fully Convolutional Networks (FCN) for Semantic Segmentation 

## Technical Formulation of Problem 

* Set up environment as described in [Project Repository](https://github.com/cwiz/RoboND-DeepLearning-Project)
* Complete [RoboND-Segmentation-Lab](https://github.com/udacity/RoboND-Segmentation-Lab)
* Get trained model to work in simulator scenario
* [Optional] Experiment with more recent models

## Mathematical Models

Semantic Segmentation Problem is defined as categorizing pixels in image into pre-defined categories. For example, in Autonomous Driving (AD) these categories might be: Road, Pedestrians, Other Cars and Buildings. In AD categoy count can be in 20-50s.

In case of a flying drone we are only dealing with 3 categories: Hero, People and Foreground.

Prior to widespread arrival of DNNs and CNNs Semantic Segmentation has been a challenging task. Current approaches such as FCNs solve Semantic Segmentation on human levels. For some applications, such as medicine FCN approach comes close to human-expert levels.

In this project I've experimented with 2 solutions: FCN-8 and VGG16-UNET. First one is discussed in lab and second one is more complex network based on [VGG16](https://neurohive.io/en/popular-networks/vgg16/).

Core idea behind FCNs is to process input via bunch of convolutional layers and then apply transpose convolutions to build another image with same dimensions as original one but each pixel belonging to one of predefined categories. Typical architectures are described in Jupyter Notebook for each solution.

### FCN-8

* [Jupyter Notebook](https://github.com/cwiz/RoboND-Segmentation-Lab/blob/master/code/segmentation_lab.ipynb)

```python
def fcn_model(inputs, num_classes):
    l1 = encoder_block(inputs, 32, 2)
    l2 = encoder_block(l1, 64, 2)
    l3 = conv2d_batchnorm(l2, 128, kernel_size=1, strides=1)
    l4 = decoder_block(l3, l1, 64)
    l5 = decoder_block(l4, inputs, 32)
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(l5)
```

### VGG16-UNET

* [Jupyter Notebook](https://github.com/cwiz/RoboND-Segmentation-Lab/blob/master/code/segmentation_lab-vgg16-unet.ipynb)
* [Reference Implementation](https://github.com/HLearning/unet_keras)
* [Original Paper](https://arxiv.org/abs/1505.04597)

```python
def vgg16_unet(input_shape=(256,256,3), weights='imagenet'):
    vgg16_model = keras.applications.VGG16(input_shape=input_shape, weights=weights, include_top=False)

    block4_pool = vgg16_model.get_layer('block4_pool').output
    block5_conv1 =  layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block4_pool)
    block5_conv2 =  layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block5_conv1)
    block5_drop = layers.Dropout(0.5)(block5_conv2)

    block6_up = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(block5_drop))
    block6_merge = layers.Concatenate(axis=3)([vgg16_model.get_layer('block4_conv3').output, block6_up])
    block6_conv1 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block6_merge)
    block6_conv2 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block6_conv1)
    block6_conv3 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block6_conv2)


    block7_up = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(block6_conv3))
    block7_merge = layers.Concatenate(axis=3)([vgg16_model.get_layer('block3_conv3').output, block7_up])
    block7_conv1 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_merge)
    block7_conv2 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_conv1)
    block7_conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_conv2)


    block8_up = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(block7_conv3))
    block8_merge = layers.Concatenate(axis=3)([vgg16_model.get_layer('block2_conv2').output, block8_up])
    block8_conv1 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block8_merge)
    block8_conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block8_conv1)

    block9_up = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(block8_conv2))
    block9_merge = layers.Concatenate(axis=3)([vgg16_model.get_layer('block1_conv2').output, block9_up])
    block9_conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_merge)
    block9_conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_conv1)

    block10_conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_conv2)
    block10_conv2 = layers.Conv2D(3, 1, activation='sigmoid')(block10_conv1)

    model = models.Model(inputs=vgg16_model.input, outputs=block10_conv2)
    return model
```

### Results

* VGG16-UNET was hard to train to get decent results. With wrong learning rate or batch size > 4 network would always predict dominating class on all pixels
* FCN-8 is surprisingly performant for it's weight (153 KB)
* VGG16-UNET was trained on full-size images (224x224) while FCN-8 processes 128x128 images
* Trained Model Weights [link](https://drive.google.com/open?id=1-k1X1IIjrWK1vunAJ9yvbGzLIYPGRax9)

#### FCN-8 Model Metrics

```
number of validation samples intersection over the union evaluated on 1184
average intersection over union for background is 0.9884829621796758
average intersection over union for other people is 0.333322344775126
average intersection over union for hero is 0.14917389097174866
global average intersection over union is 0.4903263993088502
```

#### VGG16-UNET Model Metrics

```
number of validation samples intersection over the union evaluated on 1184
average intersection over union for background is 0.9910767980300638
average intersection over union for other people is 0.4067272314461133
average intersection over union for hero is 0.15575457082402383
global average intersection over union is 0.5178528667667336
```

* Simulator [video](https://youtu.be/OMV4EAk9bng).
* Neural network output [video](https://www.youtube.com/watch?v=aP7xrh_0_5s&feature=youtu.beg).
