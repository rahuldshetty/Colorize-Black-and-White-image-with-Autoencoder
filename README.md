# Colorize-Black-and-White-image-with-Neural-Network

Colorizing old black and white images usually required human interventions. The given model uses Autoencoder , which has been trained to convert a B&W image to a RGB color image. The model is been trained for some 400 samples. You can add some more to the dataset and retrain. 

## Getting Started

1) Clone the repository into your system. 
2) I used the opencountry image dataset from http://cvcl.mit.edu/database.htm
3) Download them and place it in the apporpriate folder.
4) Use the prepareData.py file to create a data.pkl which contains the data as X(256,256,1) ,Y(256,256,3)
5) Use the program.py to train the autoencoder and save the weights as 'finalmodel.h5'
6) Use the test.py for testing some samples.

### Prerequisites

Python 3 with: Keras , OpenCv , Tensorflow 


### Installing

