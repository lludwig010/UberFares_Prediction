import pandas as pd
import torch
import time
import numpy as np
#from transformers import AutoTokenizer, AutoModel
import re
import math
import os
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns



def generate_sin_cos_representation(date_str, time_str):
    '''
    Generate sine and cosine representations for the given date and time, to preserve cyclical information within the data.

    Args:
        date_str (str): Date in the format "%m-%d" (e.g., "12-25" for December 25).
        time_str (str): Time in the format "%H:%M:%S" (e.g., "14:30:00" for 2:30 PM).

    Returns:
        list: A list containing four elements:
            - Sine of the normalized day of the year.
            - Cosine of the normalized day of the year.
            - Sine of the normalized time of the day.
            - Cosine of the normalized time of the day.
    '''

    #Convert the date and time to a datetime object
    datetime_obj = datetime.strptime(f"{date_str} {time_str}", "%m-%d %H:%M:%S")

    #Calculate day of the year (1 to 365)
    day_of_year = datetime_obj.timetuple().tm_yday

    #Calculate total seconds in the day
    total_seconds = datetime_obj.hour * 3600 + datetime_obj.minute * 60 + datetime_obj.second

    #Normalize day of the year and total seconds to a range of 0 to 1
    day_normalized = (day_of_year - 1) / 364
    time_normalized = total_seconds / 86400

    #Generate sine and cosine representations
    sin_cos_representation = [
        math.sin(2 * math.pi * day_normalized),
        math.cos(2 * math.pi * day_normalized),
        math.sin(2 * math.pi * time_normalized),
        math.cos(2 * math.pi * time_normalized)
    ]

    return sin_cos_representation

date_tensor_sin = None
date_tensor_cos = None
time_tensor_sin = None
time_tensor_cos = None

def generateDataSets(seed = None):
    """
    Generate datasets from the Uber.csv file, goes through the columns and stores the pickup and dropoff coordinates,  
    date and time of pickup, number of passengers and fare amount into the UbereDataTensor.pt file.

    Args:
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        None
    """
    df = pd.read_csv('archive/uber.csv', skiprows=1)

    data_array = df.values

    data_array_trunc = data_array[:, 1:]
   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    row_number_leap = []
    #Add the time information to the dataset
    for i, row in enumerate(data_array_trunc):

        date_and_time = row[2]

        # Use a regular expression to remove the year and ' UTC'
        modified_string = re.sub(r'\d{4}-', '', date_and_time).replace(" UTC", "")
        # Split date and time
        date_str, time_str = modified_string.split()

        try:
            date_time_data = generate_sin_cos_representation(date_str, time_str)
        except:
            row_number_leap.append(i)
            continue

        #To create the initial tensor on the first row
        if i == 0:
            date_tensor_sin = torch.tensor(date_time_data[0])
            date_tensor_cos = torch.tensor(date_time_data[1])
            time_tensor_sin = torch.tensor(date_time_data[2])
            time_tensor_cos = torch.tensor(date_time_data[3])

            date_tensor_sin = date_tensor_sin.view(1)
            date_tensor_cos = date_tensor_cos.view(1)
            time_tensor_sin = time_tensor_sin.view(1)
            time_tensor_cos = time_tensor_cos.view(1)

        #For every other row, concatenate to the original one
        else:
            new_date_tensor_sin = torch.tensor(date_time_data[0])
            new_date_tensor_cos = torch.tensor(date_time_data[1])
            new_time_tensor_sin = torch.tensor(date_time_data[2])
            new_time_tensor_cos = torch.tensor(date_time_data[3])

            new_date_tensor_sin = new_date_tensor_sin.view(1)
            new_date_tensor_cos = new_date_tensor_cos.view(1)
            new_time_tensor_sin = new_time_tensor_sin.view(1)
            new_time_tensor_cos = new_time_tensor_cos.view(1)


            date_tensor_sin = torch.cat((date_tensor_sin, new_date_tensor_sin))
            date_tensor_cos = torch.cat((date_tensor_cos, new_date_tensor_cos))
            time_tensor_sin = torch.cat((time_tensor_sin, new_time_tensor_sin))
            time_tensor_cos = torch.cat((time_tensor_cos, new_time_tensor_cos))


    data_array_trunc = np.delete(data_array_trunc, row_number_leap, axis=0)

    #reformat tensor from unecessary data in orginal csv file
    remove_time_string_array = np.delete(data_array_trunc, 2, axis=1)
    remove_key = np.delete(remove_time_string_array, 0, axis=1)

    #convert cleaned uber data into a tensor
    cleaned_data_float32 = remove_key.astype(np.float32)
    cleaned_data_tensor = torch.from_numpy(cleaned_data_float32)

    date_tensor_sin = date_tensor_sin.unsqueeze(1)

    cleaned_data_tensor = torch.cat((cleaned_data_tensor, date_tensor_sin), dim=1)

    date_tensor_cos = date_tensor_cos.unsqueeze(1)
    cleaned_data_tensor = torch.cat((cleaned_data_tensor, date_tensor_cos), dim=1)
    
    time_tensor_sin = time_tensor_sin.unsqueeze(1)
    cleaned_data_tensor = torch.cat((cleaned_data_tensor, time_tensor_sin), dim=1)

    time_tensor_cos = time_tensor_cos.unsqueeze(1)
    cleaned_data_tensor = torch.cat((cleaned_data_tensor, time_tensor_cos), dim=1)

    print(f"UberDataTensor shape: {cleaned_data_tensor.shape}")

    directory_path = "BoxPlots"

    #Create BoxPlot Directory for future vizualization
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    torch.save(cleaned_data_tensor, 'UberDataTensor.pt')


def cleanZero(dataset_path):
    """
    Clean the dataset by removing rows with NaN values or zeroes in specific columns.

    Args:
        dataset_path (str): Path to the dataset tensor file.

    Returns:
        None
    """

    data_tensor= torch.load(dataset_path)

    nan_mask = torch.isnan(data_tensor).any(dim=1)
    tensor_no_nan = data_tensor[~nan_mask]

    #None of the coordinate columns or passenger numbers or fares should be 0
    zero_first_col_mask = torch.any(tensor_no_nan[:, :6] == 0.0, dim=1)
    tensor_cleaned = tensor_no_nan[~zero_first_col_mask]

    torch.save(tensor_cleaned, 'UberDataTensor_NoNAN.pt')

def identifyOutliers(DataTensor):
    """
    Identify and filter out outliers in the dataset based on specific column values
    and save the cleaned tensor. Use box plots to visualize data distribution. Outliers here are defined as either clearly faulty 
    and nonsensical values or rides that fall outside of NYC.

    Args:
        DataTensor (str): Path to the dataset tensor file.

    Returns:
        None
    """

    DataTensor = torch.load(DataTensor)
    Coords_Price_Num_People_Tensor = DataTensor[:, 0:5]

    #Convert tensor to DataFrame
    df = pd.DataFrame(Coords_Price_Num_People_Tensor.numpy(), columns=[f'Column{i+1}' for i in range(Coords_Price_Num_People_Tensor.size(1))])

    #Create box-and-whisker plots to visualize distribution of the data so far
    plt.figure(figsize=(15, 10))
    for i in range(df.shape[1]):
        plt.subplot(1, 6, i+1)
        sns.boxplot(y=df.iloc[:, i])
        plt.title(f'Column {i+1}')

        plt.yticks(list(plt.yticks()[0]) + [df.iloc[:, i].min(), df.iloc[:, i].max()])

    plt.tight_layout()
    plt.savefig('BoxPlots/StartingDataDistribution.png')

    torch.set_printoptions(sci_mode=False, precision=6)

    #New York City Coordinate Limits To Keep Only Valid Coordiantes For Both Arrival And Departure In New York City.
    print("Applying mask to only keep coordiantes in dataset from within NYC and Fares between 0.5 and 100 dollars as well as only default (basic) option rides (cars with 3 passengers)")
    mask = (
        ((DataTensor[:, 0] < 0.5) | (DataTensor[:, 0] > 100)) |
        ((DataTensor[:, 1] < -74.2591) | (DataTensor[:, 1] > -73.7004)) |
        ((DataTensor[:, 3] < -74.2591) | (DataTensor[:, 3] > -73.7004)) |
        ((DataTensor[:, 2] < 40.4774) | (DataTensor[:, 2] > 40.9176)) |
        ((DataTensor[:, 4] < 40.4774) | (DataTensor[:, 4] > 40.9176)) |
        (DataTensor[:, 5] > 3)
    )

    #apply reverse mask to the DataTensor
    DataTensor = DataTensor[~mask]

    torch.set_printoptions(sci_mode=False, precision=6)

    print(f"Data Tensor shape after remove outliers from other columns {DataTensor.shape}")

    Coords_Num_People_Filter_Outlier_Tensor = DataTensor[:, 0:5]

    df = pd.DataFrame(Coords_Num_People_Filter_Outlier_Tensor.numpy(), columns=[f'Column{i+1}' for i in range(Coords_Num_People_Filter_Outlier_Tensor.size(1))])

    #Create box-and-whisker plot to visualize distribution after removal of outliers 
    plt.figure(figsize=(15, 10))
    for i in range(df.shape[1]):
        plt.subplot(1, 6, i+1)
        sns.boxplot(y=df.iloc[:, i])
        plt.title(f'Column {i+1}')

        plt.yticks(list(plt.yticks()[0]) + [df.iloc[:, i].min(), df.iloc[:, i].max()])

    plt.tight_layout()
    plt.savefig('BoxPlots/AfterAllFilters.png')

    torch.save(DataTensor, 'UberDataTensorCleanedZero_CleanedOutlier_NoNAN.pt')



def NormalizeAndSplit(dataset_path, dataDirSave, seed=False):
    """
    Normalize and split the dataset into training, validation, and test sets.
    Also saves the preprocessed data.

    Args:
        dataset_path (str): Path to the dataset tensor file.
        dataDirSave (str): Directory where processed data will be saved.
        seed (int, optional): Random seed for reproducibility. Default is False.

    Returns:
        tuple: A tuple containing:
            - train_set_final (torch.Tensor): Final training set features.
            - train_ground_truth (torch.Tensor): Ground truth labels for training set.
            - validation_set_final (torch.Tensor): Final validation set features.
            - val_ground_truth (torch.Tensor): Ground truth labels for validation set.
            - test_set_final (torch.Tensor): Final test set features.
            - test_ground_truth (torch.Tensor): Ground truth labels for test set.
    """
        
    data_tensor= torch.load(dataset_path)


    #Remove passenger data, all rides are only standard car, so it would not matter
    data_tensor = torch.cat((data_tensor[:, :5], data_tensor[:, 6:]), dim=1)


    df_train_set = pd.DataFrame(data_tensor.numpy())



    #Randomize the rows
    if seed:
        torch.manual_seed(seed)
    

    num_rows = data_tensor.size(0)
    shuffled_indices = torch.randperm(num_rows)
    shuffled_tensor = data_tensor[shuffled_indices]
    

    #Use as check to make sure correct cleaned dataset is used
    if torch.isnan(shuffled_tensor).any():
        print("Tensor contains Inf values")

    #split into train, val, test sets using 80-10-10
    total_rows = data_tensor.size(0)
    split_80 = int(total_rows * 0.8)
    split_10 = int(total_rows * 0.1)


    train_set = shuffled_tensor[:split_80]
    validation_set = shuffled_tensor[split_80:split_80+split_10]
    test_set = shuffled_tensor[split_80+split_10:]

    #get ground truth labels
    train_ground_truth = train_set[:, 0]
    val_ground_truth = validation_set[:, 0]
    test_ground_truth = test_set[:, 0]

    train_set_final = train_set[:, 1:]
    validation_set_final = validation_set[:, 1:]
    test_set_final = test_set[:, 1:]


    train_coordinates_num_passenger = train_set_final[:, :4]
    validation_coordinates_num_passenger = validation_set_final[:, :4]
    test_coordinates_num_passenger = test_set_final[:, :4]

    train_time = train_set_final[:, 4:]
    validation_time = validation_set_final[:, 4:]
    test_time = test_set_final[:, 4:]
    

    
    #create scaler for scaling based off of test set
    scaler = MinMaxScaler()
    scaler.fit(train_coordinates_num_passenger)

    min_vals = train_coordinates_num_passenger.min(dim=0, keepdim=True)[0]
    max_vals = train_coordinates_num_passenger.max(dim=0, keepdim=True)[0]

    
    #normalize between -1 and 1 
    train_coordinates_num_passenger_scaled = 2 * ((train_coordinates_num_passenger - min_vals) / (max_vals - min_vals)) - 1
    validation_train_coordinates_num_passenger_scaled = 2 * ((validation_coordinates_num_passenger - min_vals) / (max_vals - min_vals)) - 1
    test_train_coordinates_num_passenger_scaled = 2 * ((test_coordinates_num_passenger - min_vals) / (max_vals - min_vals)) - 1
    

    #standardize -> use if want to do standardization instead
    '''
    column_means = train_coordinates_num_passenger.mean(dim=0)
    column_stds = train_coordinates_num_passenger.std(dim=0)

    gt_min = train_ground_truth.mean(dim=0)
    gt_max = train_ground_truth.std(dim=0) 
    '''

    train_set_final_scaled = torch.cat((train_coordinates_num_passenger_scaled, train_time), dim = 1)
    validation_set_final_scaled = torch.cat((validation_train_coordinates_num_passenger_scaled, validation_time), dim = 1)
    test_set_final_scaled = torch.cat((test_train_coordinates_num_passenger_scaled, test_time), dim = 1)
    
    print("Split Shapes")
    print("-------------")
    print(f"Train tensor shape: {train_set_final_scaled.shape}")
    print(f"Validation tensor shape: {validation_set_final_scaled.shape}")
    print(f"Test tensor shape: {test_set_final_scaled.shape}")


    visualize_train = train_set_final_scaled

    df = pd.DataFrame(visualize_train.numpy(), columns=[f'Column{i+1}' for i in range(visualize_train.size(1))])

    #Create box-and-whisker plots for visualization of distribution of train set
    plt.figure(figsize=(15, 10))
    for i in range(df.shape[1]):
        plt.subplot(1, 9, i+1)
        sns.boxplot(y=df.iloc[:, i])
        plt.title(f'Column {i+1}')

        plt.yticks(list(plt.yticks()[0]) + [df.iloc[:, i].min(), df.iloc[:, i].max()])

    plt.tight_layout()
    plt.savefig('BoxPlots/TrainSet_DistributionAfterPreprocessing.png')

    os.makedirs(f'dataFolder/{dataDirSave}', exist_ok=True)
    
    torch.save(train_ground_truth, f'dataFolder/{dataDirSave}/train_ground_truth.pt')
    torch.save(val_ground_truth, f'dataFolder/{dataDirSave}/val_ground_truth.pt')
    torch.save(test_ground_truth, f'dataFolder/{dataDirSave}/test_ground_truth.pt')
    torch.save(train_set_final_scaled, f'dataFolder/{dataDirSave}/train_set_final.pt')
    torch.save(validation_set_final_scaled, f'dataFolder/{dataDirSave}/validation_set_final.pt')
    torch.save(test_set_final_scaled, f'dataFolder/{dataDirSave}/test_set_final.pt')
    
    
    column_names_train = [['pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'date_sin', 'date_cos', 'time_sin', 'time_cos']]
    df_train_set = pd.DataFrame(train_set_final_scaled.numpy(), columns=column_names_train)
    

    return train_set_final, train_ground_truth, validation_set_final, val_ground_truth, test_ground_truth, test_set_final

if __name__ == "__main__":

    start_time = time.time()

    print("Starting Generation Of Dataset From archive/uber.csv...")
    generateDataSets()
    print("Generation Of Dataset Complete")

    print("Starting Preprocessing...")
    print("1. Cleaning all NAN and all 0 rows")
    cleanZero('UberDataTensor.pt')

    print("2. Filtering Non New York Coordinate Outliers")
    identifyOutliers('UberDataTensor_NoNAN.pt')

    print("3. Shuffle, Split, Normalize (Based Off Of train) Datasets")
    #generate normalized datasets
    dataDirSave = 'ProcessedData'
    train_set_final, train_ground_truth, validation_set_final, val_ground_truth, test_groud_truth, test_set_final = NormalizeAndSplit('UberDataTensorCleanedZero_CleanedOutlier_NoNAN.pt', dataDirSave)

    print("Preprocessing Finished")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the total execution time
    print(f"Total time to generate preprocessed dataset: {elapsed_time:.2f} seconds")