# FIGURE OUT WHY I NEED TO DOUBLE-IMPORT !!!
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
import os
import seaborn as sns
import matplotlib.pyplot as plt

###########################################################################################################################################

def preprocess_Aleph(data_path, test_frac=0.15, info=False):
    """
    Loads the chosen Aleph dataset. Keeps only 6 predictors (see last exercise session).
    Splits into train/valid/test sets, and resets their indices.
    
    INPUT
    data_path : Relative path to the Aleph data .csv file.
    test_frac : Fraction of all observations that will be test set. (test-size = val-size.)
    info      : Boolean. If true, prints information about loaded data.
    
    RETURNS: Unscaled, unnoised dataframes.
    
    X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    val_frac  = test_frac/(1-test_frac)
    
    data = pd.read_csv(data_path)
    if info:
        data.info(verbose=True)
    
    # I only use 6 predictors, like the original Aleph network used.
    target = data['isb']
    predictors = data[['prob_b', 'spheri', 'pt2rel', 'multip', 'bqvjet', 'ptlrel', 'nnbjet']]

    # "Full" means the dataset contains the "nnbjet" variable, which we shouldn't use.
    # nnbjet is the predictions given by Aleph.
    X_rest, X_test_full, y_rest, y_test = train_test_split(predictors, target, test_size = test_frac)
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_rest, y_rest, test_size = val_frac)
    
    # Reset indices of all Pandas Dataframes and Series. This prevents headaches later.

    X_train_full = X_train_full.reset_index(drop="True")
    X_valid_full = X_valid_full.reset_index(drop="True")
    X_test_full  = X_test_full.reset_index(drop="True")
    y_train = y_train.reset_index(drop="True")
    y_valid = y_valid.reset_index(drop="True")
    y_test  = y_test.reset_index(drop="True")
    
    # 'nnbjet' is Aleph's predictions. Drop them from training sets.
    X_train_unscaled  = X_train_full.drop('nnbjet', axis=1)
    X_valid_unscaled  = X_valid_full.drop('nnbjet', axis=1)
    X_test_unscaled   = X_test_full.drop('nnbjet', axis=1)
    
    return X_train_unscaled, y_train, X_valid_unscaled, y_valid, X_test_unscaled, y_test

###########################################################################################################################################

def scale_data(X_train_unscaled, lodfs):
    """
    Instantiates StandardScaler and fits it to X_train_unscaled. Then, scales all dataframes in lodfs.
    
    RETURNS: List of scaled dataframes, in same order as they appear in lodfs.
    """
    
    # Create the standard scaler instance.
    scaler = StandardScaler()
    scaler.set_output(transform='pandas')
    scaler.fit(X_train_unscaled)
    
    # Scale each dataframe in lodfs.
    scaled_dfs = []
    for df in lodfs:
        df_scaled = scaler.transform(df)
        scaled_dfs.append(df_scaled)
        
    return scaled_dfs

############################################################  Noise functions  ############################################################

def add_noise(df, lon):
    """
    Adds noise to each predictor in df, using custom noise functions.
    df  : Dataframe to add noise to.
    lon : List (size = # of columns) of noise functions to apply to each column.
          NOTE: If noise function looks like noise(...), lon can be [noise].
    OUTPUT: Pandas dataframe of the processed data.
    """
    columns = df.columns
    array = df.to_numpy()
    array_shape = array.shape
    modified_array = np.copy(array) # This is where the modifications will go.
    
    for i in range(array_shape[1]): # Iterate over the columns.
        noise_fcn = lon[i]
        modified_array[:,i] = noise_fcn(modified_array[:,i])
        
    modified_df = pd.DataFrame(modified_array, columns=columns)
    
    return modified_df

def noise_tester(df, df_noised, histogram=True):
    columns = df.columns
    difference = df_noised - df
    for column in columns:
        if histogram:
            pred_noise = difference[column]
            plt.hist(pred_noise, bins=50)
            plt.xlabel(f'Noise in {column}')
            plt.show()
        else:
            plt.plot(df[column], df_noised[column], 'o', markersize=2, label=column)
            plt.xlabel("Simulated")
            plt.ylabel("Real")
            plt.legend()
            plt.show()
            
def gaussian_noise(scale=1, s_mean=0, s_std=1): # X is a predictor column.
    def g(X=np.ones(1), scale=scale, s_mean=s_mean, s_std=s_std):
        std  = np.std(X)
        mean = np.mean(X)
        noise = np.random.normal(mean*s_mean, std*s_std, len(X))
        X_new = (X + noise) * scale
#         plt.hist(noise, bins=50)
        return X_new
    
    return g

def sin_noise(t=0.3,n=3):
    def s(X=np.ones(1), t=t, n=n):
        std  = np.std(X)
        mean = np.mean(X)
        max_ = np.max(X)
        noise = 1/t*mean*np.sin(2*np.pi*n/max_ * np.copy(X))
        X_new = X + noise
        return X_new
    
    return s

###########################################################################################################################################

def df_to_device(lodf, device, dtypes=None):
    """
    Converts each Dataframe in the list (lodf) to a Tensor, then
    sends it to the device. Returns the list of tensors on device.
    """
    # Initiate list that will contain tensors on target device.
    lotensor = []
    for i, df in enumerate(lodf):
        # Choose dtype of the i-th tensor.
        if dtypes != None:
            print("Dtypes were specified.")
            dtype = dtypes[i]
        else:
            dtype = torch.float32
        
        # Convert pandas df to tensor.
        tensor = torch.tensor(df.to_numpy(), dtype=dtype)
        # Send tensor to device.
        tensor = tensor.to(device)
        # Save to list of tensors.
        lotensor.append(tensor)
    
    return lotensor

###########################################################################################################################################

def make_batches(X_tensor, y_tensor, num_batches, batch_size):
    """
    Accepts predictor and target tensors (which must have same # rows).
    Shuffles them together, then returns batches for each in the form of
    lists.
    """
    # Check if tensors have correct shapes:
    if X_tensor.shape[0] != y_tensor.shape[0]:
        print("ERROR! Shapes are incorrect!")
        1/0
    
    # Shuffle tensors by index slicing. NOTE: I think operations are happening on GPU.
    indices = torch.randperm(X_tensor.shape[0]) # ??? I still don't know how this works precisely.
    X       = torch.clone(X_tensor)[indices]
    y       = torch.clone(y_tensor)[indices]
    
    # Now, make batches. NOTE: Slicing beyond end of tensors causes no issues!
    X_list = [X[i*batch_size:(i+1)*batch_size,:] for i in range(num_batches)]
    y_list = [y[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    return X_list, y_list

###########################################################################################################################################

def train_one_epoch(model, loss_fn, optimizer, X_batch_list, y_batch_list, epoch=1, num_epochs=1):
    # Set the model to training mode
    model.train()
    # Loop over batches.
    running_loss = 0
    num_batches = len(X_batch_list)
    for i in range(num_batches):
        X_batch = X_batch_list[i]
        y_batch = y_batch_list[i]
        
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        train_outputs = model(X_batch)
        # Compute loss
        train_loss = loss_fn(train_outputs, y_batch.unsqueeze(1))
        # Back propagation and weight updates.
        train_loss.backward()
        optimizer.step()
        
        # Add loss to running loss.
        running_loss += train_loss.item()
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    running_loss = running_loss/num_batches # Is this correct???
    return running_loss

###########################################################################################################################################

def validate_one_epoch(model, loss_fn, X_valids, y_valids): # Include batching later? For now, validation size is small.
    # Set the model to evaluation mode.
    model.eval()
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        valid_outputs = model(X_valids)
        valid_loss = loss_fn(valid_outputs, y_valids.unsqueeze(1))
    return valid_loss.item()

###########################################################################################################################################

def train_model(model, t_X_train, t_y_train, t_X_valid, t_y_valid, loss_fn,
                optimizer, num_epochs, batch_size, num_batches, freq, eps):
    
    # Everything below here is the training sequence/loop.
    
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(num_epochs):
        print("Epoch number: {}".format(epoch))
        # Make batches.
        X_batch_list, y_batch_list = make_batches(t_X_train, t_y_train, num_batches, batch_size)

        # Training step.
        train_loss = train_one_epoch(model, loss_fn, optimizer, X_batch_list, y_batch_list, epoch=epoch, num_epochs=num_epochs)
        # Validation step. Happens less frequently!
        if epoch % freq == 0:
            # Note that torch.no_grad() is called within validate_one_epoch().
            valid_loss = validate_one_epoch(model, loss_fn, t_X_valid, t_y_valid)
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
                
    # Return the trained model, as well as training and test losses.
    return model, train_loss_list, valid_loss_list

###########################################################################################################################################

def plot_training_losses(freq, train_loss, valid_loss, losstype='BCE loss'):
    epoch = np.arange(len(train_loss))
    epoch[1:] = freq * epoch[1:]
    plt.plot(epoch, (train_loss), 'o', markersize=3, label='Training loss')
    plt.plot(epoch, (valid_loss), 'o', markersize=3, label='Valid loss')
    plt.xlabel("Epoch count")
    plt.ylabel(losstype)
    plt.legend()
    plt.show()

###########################################################################################################################################

def clf_accuracy(model, X_test, y_test):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        outputs = model(X_test)
        probs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
        preds = (probs > 0.5).float()   # Convert probabilities to binary predictions (0 or 1)
        correct += (preds.squeeze() == y_test).sum().item()  # Compare predictions with true labels
        total += y_test.size(0)  # Accumulate the total number of labels

    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

###########################################################################################################################################

def make_roc(model, lotuples, save = False, directory=0, filename = 'roc_curve'):
    """
    INPUTS
    model    : Trained model.
    lotuples : List of (X, y, label) tuples. X = predictors, y = labels, label = legend on graph.
    save     : If True, save plot. Else, don't.
    OUTPUTS
    Returns nothing. Plots (and potentially saves) ROC curves.
    """
    with torch.no_grad(): # Disable gradient calculation for evaluation
        plt.figure()
        for X, y, label in lotuples:
            outputs = model(X)
            predictions = torch.sigmoid(outputs).squeeze()  # Apply sigmoid to get probabilities
            fpr, tpr, _ = roc_curve(y.cpu().numpy(), predictions.cpu().numpy())
            roc_auc = auc(fpr, tpr)

    #     # Save ROC data
    #     np.save(ourdir + f'/ROC_{varReduction}.npy', np.array([fpr, tpr]))

            # Plot ROC curve
            plt.plot(fpr, tpr, 'o', markersize=1, lw=2, label=f'{label} (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    if save and directory != 0:
        plt.savefig(f'{directory}/{filename}.png')
    plt.show()

###########################################################################################################################################



###########################################################################################################################################



















