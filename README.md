#  Introduction
This project attempts to predict NYC Uber ride fare costs givencoordinates of ride dropoffs and pickups as well as the time of the pickup from the dataset at https://www.kaggle.com/datasets/yasserh/uber-fares-dataset. To solve
this regression task, I preprocessd and loaded the data into a custom model and training script built with PyTorch and core machine learning concepts. During data preprocessing, in addition to cleaning the data, standardizing and running train, validation
and test splits, I transformed date and time data into sin and cos representations to keep the cylcical structure of the information while making it more manageable for the model to efficiently learn. One of the main goals for this project
was for easy optimization of the training process as well as visualizing different affects on model performance. Therefore, hidden layer parameters, hyperparameters and different training and regulariazation factors can be aeasily added or change w
with command line arguments built within the project. To measure performance and visualize the bias-variance tradeoff, to prevent over or under fitting, training and validation curves are plotted at the end of training as well as the aparameters that were used for analysis. 

#  Running The Process 
To start the process, we first need to download and preprocess the dataset using preprocess.py. By default, the train, validation and test sets should appear in the 'ProcessedData' directory. Visualization of data distributions preprocessing 
procedure are placed in the 'BoxPlot' directory. After the data is processed, running training.py will initiate the training loop/ The script takes several arguments to modify certain hyperparameters where each command is detailed within the file.
Trainign outputs can be found in the 'TrainingOutput' directory where each run is organized by dateand time when it started.

# Data Visualization
Within the BoxPlot directory there will be three plots for data visualization. The first plot, 'StartingDataDistribution' has the distribution for the pickup coordinates. Column 1 is the price of the fare, column 2 is the latitude of the pickup,
column 3 is the longitude of the pickup, column 4 is the latitude of the dropoff while column 5 is the longitude of the dropoff. In the second plot 'AfterAllFilters' each column has the same rpresentations after filters were applied to keep coordinates 
within NYC, removing nonsensical and noisy coordinates. Finally, 'TrainSet_DistributionAfterPreprocessing' plot is the final distribution of all inputs that will be put into the model. Column 1-2 are the latitude and longituded of the pickup while Column 3-4 are 
the latitude and longitude of the dropoff. Column 5-6 are the sin and cos representations of the date, while column 7-8 are the sin and cos representations of the time. 
