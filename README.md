# CNN_MNIST
Build and optimize a CNN to recognize handwritten digits. 

The objective of this work is to explore more about picture reorganization with Convolutional Neural Networks (CNN). The dataset used in this case study is the famous MNIST (http://yann.lecun.com/exdb/mnist/) which is extremely popular to learn how to benchmak a CNN model. 

This repository is composed by 2 notebooks and 1 python script:
-	‘Save_dataSetAsH5_train_val_test.ipynb’ extract, normalized, split and store the MNIST data into .h5 files.
-	‘CNN_exploration_eval.nbconvert.ipynb’ is the main notebook where the data is loaded, and the CNN models are evaluated. To evaluate the model architecture, I performed a grid search to find the best parameter values. To finish the best model (highest accuracy value) was used to simulate a production evaluation under unknown data. The model reaches a 99.2% accuracy over 5,000 digits. 
-	‘Pwk.py’ is a python script with useful functions. 
![image](https://user-images.githubusercontent.com/82883812/148738840-2d4cb3b1-d837-459f-a422-4affecb87b1d.png)

The optimization was performed over only 32 models because of the GPU limitation of my machine (Macbook Pro - M1, 2020)

