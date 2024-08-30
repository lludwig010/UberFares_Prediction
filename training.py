from Model import MLP
import torch.nn as nn
import torch.optim as optim
import torch
import time
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import argparse
import json

def train(model,
          loss_f,
          optimizer,
          train_data,
          val_data,
          train_gt,
          val_gt,
          n_epoch=500,
          batch_size=256,
          seed_value=None,
          device='GPU',
          dir_name='TrainingOutputs',
          initial_weights = None):
    """
    Trains a simple MLP Neural Network on the Preprocessed Uber NYC rides dataset. Plots the train and validation loss as training progresses
    to measure performance.

    Asgs:
    -----------
    model (torch.nn.Module): The neural network model to be trained. Architecture can be modified in Model.py
    loss_f (torch.nn.modules.loss._Loss): The loss function to be used for training.
    optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's weights.
    train_data (torch.Tensor): The training data.
    val_data (torch.Tensor): The validation data.
    train_gt (torch.Tensor): The ground truth labels for the training data.
    val_gt (torch.Tensor): The ground truth labels for the validation data.
    n_epoch (int, optional): The number of epochs to train the model. Default is 500.
    batch_size (int, optional): The number of samples per batch. Default is 500.
    seed_value (int, optional): The seed value for reproducibility. If provided, it sets the random seed. Default is None.
    device (str, optional): The device to use for training ('CPU' or 'GPU'). Default is GPU.
    dir_name (str, optional): The directory name where the training outputs, including model checkpoints, will be saved. Default is TrainingOutputs.

    Returns:
    --------
    None
        The function saves the training and validation loss plots, and the best model's checkpoint to the specified directory.
    """
    
    total_loss_train = []
    total_loss_val = []
     
    if seed_value is not None:
        torch.manual_seed(seed_value)
    
    lenTrainDataset = len(train_data)

    if device == 'GPU':
        print("Using GPU")
        model = model.to('cuda')
        train_data = train_data.to('cuda')
        train_gt = train_gt.to('cuda')

        val_data = val_data.to('cuda')
        val_gt = val_gt.to('cuda')
    
    
    train_data = train_data.to(torch.float)
    train_gt = train_gt.to(torch.float)

    print("Train data shape and gt shape")
    print(train_data.shape)
    print(train_gt.shape)

    #checkpoint information
    bestValPerformanceLoss = None
    checkpoint_path = f'{dir_name}/nn_checkpoint.pth'

    #how many epochs to wait before early stopping is triggered for poor performance
    early_stopping_patience = 100
    early_stopping_counter = 0 

    for i in range(n_epoch):
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

        print(f"Epoch {i + 1}")
        index = 0
        batchNumber = 0

        loss_per_batch = []
        model.train()
        while index < lenTrainDataset:
            #Load in the data

            #case if the last batch is a little larger than the batch_size
            if lenTrainDataset - (index + batch_size) < batch_size:
                train_batch = train_data[index:lenTrainDataset]
                train_gt_batch = train_gt[index:lenTrainDataset]
            #normal case where every batch is expected batch size
            else:
                train_batch = train_data[index:index + batch_size]
                train_gt_batch = train_gt[index:index + batch_size]

            train_batch_float = train_batch
            train_gt_batch_float = train_gt_batch

            output = model.forward(train_batch_float)
            output_reformat = output.squeeze()

            loss = loss_f(output_reformat, train_gt_batch_float)
            
            '''
            For debugging
            if index == 0:
                print("________________________")         
                print(output_reformat[:5])
                print(train_gt_batch_float[:5])
                print(loss)
            '''

            loss_per_batch.append(loss)

            #set gradients back to 0 before backpropagating
            optimizer.zero_grad()
            
            #backpropagate
            loss.backward()
            #update parameters
            optimizer.step()
        
            index += batch_size + 1
            batchNumber += 1
        
        average_loss = sum(loss_per_batch) / len(loss_per_batch)
        total_loss_train.append(average_loss.cpu().item())

        print(f"average loss for epoch: {average_loss}")

        model.eval()
        with torch.no_grad():
        #get validation metrics

            val_data_float = val_data
            val_gt_float = val_gt


            outputVal = model.forward(val_data_float)
            output_val_reformat = outputVal.squeeze()

            lossVal = loss_f(output_val_reformat, val_gt_float)

            total_loss_val.append(lossVal.cpu().item())

        print(f"validation loss: {lossVal}")

        #if val loss is best, checkpoint model
        if bestValPerformanceLoss is None or lossVal < bestValPerformanceLoss:
            bestValPerformanceLoss = lossVal
            torch.save(model.state_dict(), checkpoint_path)
            print("Taking checkpoint, better performance")

            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
    
    xAxisEpoch = np.linspace(1, len(total_loss_train), len(total_loss_train))
    plt.plot(xAxisEpoch, total_loss_train, label='Train Loss')
    plt.plot(xAxisEpoch, total_loss_val, label='Validation Loss')

    #plot loss during training
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=16)
    plt.title("Validation And Train Loss Over Epochs")
    plt.legend(loc='upper right', title='Legend', fontsize='medium')

    plt.savefig(f'{dir_name}/Loss_Plot.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP model on Uber fares NYC dataset")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--dropout_prob', type=float, default=0.25, help='Dropout probability for the model')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[36, 24, 12, 6], help='Sizes of hidden layers')
    parser.add_argument('--n_epoch', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--device', type=str, default='GPU', help='Device to use for training (CPU or GPU)')
    parser.add_argument('--data_dir', type=str, default='dataFolder/ProcessedData', help='Directory to use for training data')
    parser.add_argument('--warm_start', type=str, default=None, help='Provide the path to checkpoint weights for warm start')

    args = parser.parse_args()

    # Get current date and time
    now = datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = "'TrainingOutputs/Training_" + formatted_date_time
    os.makedirs(dir_name, exist_ok=True)
    
    # Save arguments to a JSON file for reproducibility
    args_dict = vars(args)
    with open(f'{dir_name}/run_arguments.json', 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    DataDir = args.data_dir
    train_set_final = torch.load(f'{DataDir}/train_set_final.pt')
    train_ground_truth = torch.load(f'{DataDir}/train_ground_truth.pt')
    validation_set_final = torch.load(f'{DataDir}/validation_set_final.pt')
    val_ground_truth = torch.load(f'{DataDir}/val_ground_truth.pt')
    test_ground_truth = torch.load(f'{DataDir}/test_ground_truth.pt')
    test_set_final = torch.load(f'{DataDir}/test_set_final.pt')

    print("Initializing model, optimizer, and loss function")
    model = MLP(8, args.hidden_layers, 1, dropout_prob=args.dropout_prob)
    #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    if args.warm_start:
        try:
            print(f"Loading model weights from {args.warm_start}")
            checkpoint = args.warm_start
            print(checkpoint)
            readCheckpoint = torch.load(checkpoint)
            print(type(readCheckpoint))
            model.load_state_dict(readCheckpoint)

        except Exception as e:
            print(f"Error loading the checkpoint file: {e}")
            exit(1)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_f = nn.HuberLoss()
    #MSE loss can be used instead, however HuberLoss is a good middle ground between L1 and L2
    #loss_f = nn.MSELoss()

    print("Training")
    train(model, loss_f, optimizer, train_set_final, validation_set_final, train_ground_truth, val_ground_truth, n_epoch=args.n_epoch, batch_size=args.batch_size, device=args.device, dir_name=dir_name)