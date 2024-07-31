![overview](https://i.ibb.co/jRwSWcQ/POWERPNT-1r-JXWKz-Jxe.png)

[[Referred Paper](https://arxiv.org/abs/2204.03355)] [[Demo Youtube](https://youtube.com/shorts/y_dw6AWiR-8)]

This repository contains the official code from __Pedal Keeper(페달키퍼). Sudden Unintended Acceleration Prevention System__. 

Pedal Keeper is an AI-based driver assistance device that prevents sudden jerking accidents in vehicles. It is known that sudden jerking accidents are mainly caused by (i) defects in the pedal sensing system and (ii) user misoperation. To this end, in PedalKeeper, Vision AI analyzes the surrounding situation to predict the pedal sensor output and compare it with the acquired pedal sensor value to detect sudden jerks. However, general vision has limited practicality in real-world vehicle driving environments due to the limitations of data acquired in low light, motion blur, and high-contrast situations (e.g., backlighting). Pedal Keeper addresses this by adopting an Event Camera sensor that only detects brightness changes and a Transformer structure that has shown excellent vision classification performance in ViT. We improve the Transformer structure to better utilize the sparse Event camera data, and conduct a proof of concept using an external dataset (Event Camera Driving Dataset, EZH) and commercial pedal hardware.


### REPOSITORY REQUIREMENTS

The present work has been developed and tested with Matlab R2024a
To reproduce our results we suggest to add add-ons and use options as follows.

```MATLAB
add on
  Natural-Order Filename Sort v3.4.6 by Stephen23 
  [[link](https://kr.mathworks.com/matlabcentral/fileexchange/47434-natural-order-filename-sort)]
option
  options = trainingOptions("adam", ...
    Plots="training-progress", ...
    InputDataFormats=["SBTC","SBTC"], ...
    TargetDataFormats="CBT", ...
    MaxEpochs=json.training_params.max_epochs, ...
    InitialLearnRate=json.optim_params.optim_params.lr, ...
    MiniBatchSize=json.data_params.batch_size, ... 
    CheckpointPath="chk", ...
    Shuffle="never", ...
    ValidationData=ds_val, ...
    ExecutionEnvironment="gpu", ...
    Metrics= accuracyMetric(ClassificationMode = "multilabel"), ...
    Verbose=false)
```


### DATA DOWNLOAD

The datasets involved in the present work must be downloaded from their source and stored under a `./data` path:
 - Event Camera Dataset : [Google Drive](https://drive.google.com/file/d/1swbgP0ikgkwS97th2Z3mIjSD5kVLdZv6/view?usp=sharing)

We want to appreciate ETH offering the original open-source dataset, [Driving Event Camera Dataset (Samsung DVS Gen3)](https://rpg.ifi.uzh.ch/event_driving_datasets.html).

###  PRE-PROCESSING
Changed ros-bag files into txt files. And txt files was changed to pckl which contains 20 chunks(The events which are at same time) at each files. After building pckl, pckl were combined with labels and changed into mat by `pckl2mat.m`. And then, the files' dir were listed and saved to `./data/data_labels.m`, by `mat2ds.m`. labels were created by us with following steps.

### DATA LABELING
pckl files were converted to images and palyed like stop motion with openCV.
We create labels with logitech G29's pedal by our selves with playing images as video. Each label was created for each chunk.



### EvT INFERENCE

The inference of our models can be performed by executing: `inference.m`
At the `load` you can select the data to compute. Inference result shows 3 vectors. You can use argmax to find final result.



### EvT TRAINING

The training of a new model can be performed by executing: `trainer.m`
You can change some setting of models by change `all_params.json`. 
At the variable `options` you can select the options of training.
Note that, since the involved datasets do not contain many training samples and there is data augmentation involed in the training, final results might not be exactly equal than the ones reported in the article. If so, please perform several training executions.
