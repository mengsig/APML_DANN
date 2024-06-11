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

A = 6  # Want figures to be A6
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 35.61 * .5**(.5 * A)])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 36})
sns.set(font_scale = 1)
varReduction = 1

varReduction = 1
# Data Loading and Preprocessing
data_path = 'AlephBtag_MC_train_Nev500000.csv'
data = pd.read_csv(data_path)
print(data.describe())


data_test_path = 'AlephBtag_MC_train_Nev500000.csv'
data_test = pd.read_csv(data_path)
ourdir = f'temp_roc_mlp_transform_sin/{varReduction}'
os.makedirs(ourdir, exist_ok = True)
# Selecting features from the Initial Project
#selected_features = pd.read_csv('Classification_MarcusEngsig_PyTorchNN1_VariableList_final.csv').T
#selected_features = pd.Index(np.array(selected_features).flatten())

columns = data.columns
X = data.copy()
X = X.drop(columns = columns[-1])
y = data[columns[-1]]

X_test = data_test.copy()
X_test = X_test.drop(columns = columns[-1])
y_test = data_test[columns[-1]]
# Train/Test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Transforming simulated into real data by performing a perturbation scaling with 'mean/varReduction'
dim = X_train.shape[1]
length = X_train.shape[0]
val_length = X_val.shape[0]

#varReduction = 50  # Change this parameter to change the 'noise' in the real dataset
#sim_data = np.array(X_train.copy())
#real_data = np.array(X_train.copy())
#X_val_real = np.array(X_val.copy())
#X_test = np.array(X_test)
#
#for i in range(dim):
#    sim_data = (1+np.random.uniform(0,1, sim_data.shape)/varReduction)*sim_data.copy() + np.random.normal(0, sim_data.std() / varReduction, sim_data.shape)
#    real_data = (1+np.random.uniform(0,1, real_data.shape)/varReduction)*real_data.copy() + np.random.normal(0, real_data.std() / varReduction, real_data.shape)
#    X_test[:, i] += np.random.normal(0, np.abs(X_test[:,i].std() / varReduction), X_test.shape[0])

# Showing the difference in simulated and real data
sim_labels = y_train.copy()
real_labels = y_train.copy()

#varReduction = 1
def ourTransform(ourTensor, noise_type, varReduction):
    val = 1 / varReduction
    ourNewTensor = np.array(ourTensor)
    ourMaxes = ourNewTensor.max(axis=0)
    ourNewTensor = ourNewTensor / ourMaxes
    if noise_type == 1:
        ourNewTensor = ourNewTensor + val * np.sin(ourNewTensor * 2 * np.pi)
    elif noise_type == 2:
        ourNewTensor = ourNewTensor + val * np.cos(ourNewTensor * 2 * np.pi)
    elif noise_type ==3:
        ourNewTensor = ourNewTensor + val * ((np.sin(ourNewTensor * 2 * np.pi))**5)
    else:
        ourNewTensor = ourNewTensor*(1+val)
    varFactor = 25
    ourNewTensor  = ourNewTensor*(1+val) + np.random.normal(0, np.abs(ourNewTensor)/ (varReduction*varFactor), ourNewTensor.shape)
    ourNewTensor = ourNewTensor * ourMaxes
    return ourNewTensor
sim_data = np.array(X_train.copy())
real_data = np.array(X_train.copy())
X_val_real = np.array(X_val.copy())
#sim_data = ourTransform(sim_data, 2, varReduction = varReduction)
real_data = ourTransform(X_train, 1, varReduction = varReduction)
X_test = ourTransform(X_val, 1, varReduction = varReduction)
#X_test = X_val.copy()
#real_data = 1*(1+np.random.uniform(0,1, real_data.shape)/varReduction)*real_data.copy() + np.random.normal(0, real_data.std(axis=0) / varReduction, X_train.shape)
#real_data = sim_data.copy() + np.random.normal(0, real_data.std(axis=0) / varReduction, X_train.shape)
#X_val_real = 1*(1+np.random.uniform(0,1,X_val.shape)/varReduction)*X_val_real.copy() + np.random.normal(0, X_val_real.std(axis=0) / varReduction, X_val.shape)
#X_test = 1*(1+np.random.uniform(0,1,X_val.shape))*X_test.copy() + np.random.normal(0, X_test.std() / varReduction, X_test.shape)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.xlabel('Simulated Data, $X$', fontsize = 18)
plt.ylabel('Noised Data, $f_i(X)$', fontsize = 18)
plt.title(f'Noise $t = {varReduction}$', fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.scatter(sim_data[::,0]/sim_data[::,0].max(), real_data[::,0]/sim_data[::,0].max(), label = '$f_3(X) = X + \\frac{1}{t}\\sin(2\\pi X)$')
plt.scatter(sim_data[::,0]/sim_data[::,0].max(), sim_data[::, 0]/sim_data[::,0].max(), label = '$f_4(X) = X$', color = 'red')
plt.legend()
plt.tight_layout()
fig.savefig(f'mlp_noise_{varReduction}.pdf')
fig.savefig(f'mlp_noise_{varReduction}.png')
plt.show()

# Impute and scale the data (imputing not necessary)
# However, scaling is very important for Neural Nets
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_val_real_imputed = imputer.transform(X_val_real)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_val_real_scaled = scaler.transform(X_val_real_imputed)
sim_data_scaled = scaler.transform(sim_data)
real_data_scaled = scaler.transform(real_data)
X_test_scaled = scaler.transform(X_test_imputed)

# Converting to torch.cuda.tensors
sim_data = torch.tensor(sim_data_scaled, dtype=torch.float32).cuda()
real_data = torch.tensor(real_data_scaled, dtype=torch.float32).cuda()
X_val = torch.tensor(X_val_scaled, dtype=torch.float32).cuda()
X_val_real = torch.tensor(X_val_real_scaled, dtype=torch.float32).cuda()
sim_labels = torch.tensor(sim_labels.values, dtype=torch.long).cuda()
real_labels = torch.tensor(real_labels.values, dtype=torch.long).cuda()
y_val = torch.tensor(y_val.values, dtype=torch.long).cuda()
X_test = torch.tensor(X_test_scaled, dtype=torch.float32).cuda()
y_test = torch.tensor(y_test.values, dtype=torch.long).cuda()

# Concatenating the validation set
X_val_combined = torch.cat((X_val, X_val_real), dim=0)
y_val_combined = torch.cat((y_val, y_val), dim=0)  # Assuming y_val_real is the same as y_val

# Defining our MLP architecture
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adv_hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

        #self.adv_fc2 = nn.Linear(hidden_dim, 1)
        # Main task output layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, int(hidden_dim/2))
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(int(hidden_dim/2), output_dim)
        #self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))

        # Adversarial network forward pass
        #adv_hidden = self.relu(self.adv_fc1(hidden))
        #adv_output = self.adv_fc2(hidden)

        # Main task output
        hidden = self.relu(self.fc3(hidden))
        hidden = self.relu(self.fc4(hidden))
        #hidden = self.relu(self.fc3(hidden))
        main_output = self.fc5(hidden)

        return main_output

# Define dimensions - Change here for different structures --> now it is 20:20:20
input_dim = X_train.shape[1]
hidden_dim = 10
output_dim = 1
adv_hidden_dim = 10

# Initiating the MLP
model = MLP(input_dim, hidden_dim, output_dim, adv_hidden_dim).cuda()

# Defining loss and optimizer
criterion_main = nn.BCEWithLogitsLoss()
criterion_consistency = nn.MSELoss()
patience = 10
best_val_loss = float('inf')
counter = 0
eps = 2
num_epochs = 20000
batch_size = 64
optimizer = optim.Adam(model.parameters(), lr= eps*(10**(-3)), weight_decay=eps*(10**(-3))/num_epochs)  # Added weight_decay for L2 regularization


zeros = torch.zeros(sim_data.shape[0], 1).cuda()
ones = torch.ones(real_data.shape[0], 1).cuda()
ourBestLoss = 0

supervised = []
domain = []
combined = []

# Training
for epoch in range(num_epochs):
    model.train()

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass for the real and simulated data
    outputs_sim = model(sim_data)
#    with torch.no_grad():
#        outputs_real = model(real_data)

    # Compute losses
    main_loss_sim = criterion_main(outputs_sim, sim_labels.unsqueeze(1).float())
    #adv_loss_sim = criterion_adv(adv_outputs_sim, zeros)

    #adv_loss_real = criterion_adv(adv_outputs_real, ones)
    #adv_loss_combined = (adv_loss_sim + adv_loss_real) / 2

    # Consistency loss
    #consistency_loss = criterion_consistency(outputs_sim, outputs_real)

    # Total loss
    total_loss = main_loss_sim
    total_loss.backward()
    optimizer.step()

    supervised.append(total_loss.item())

    # Validation (just for print)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion_main(val_outputs, y_val.unsqueeze(1).float())

        val_preds = torch.sigmoid(val_outputs).cpu().numpy() > 0.5
        #val_preds = val_outputs.cpu().numpy() > 0
        val_accuracy = (val_preds.flatten() == y_val.cpu().numpy()).mean()

        test_outputs = model(X_test)
        test_loss = criterion_main(test_outputs, y_val.unsqueeze(1).float())

        test_preds = torch.sigmoid(test_outputs).cpu().numpy() > 0.5
        #test_preds = test_outputs.cpu().numpy() > 0
        test_accuracy = (test_preds.flatten() == y_val.cpu().numpy()).mean()
        if val_accuracy > ourBestLoss:
            ourBestLoss = test_accuracy
            np.save(ourdir + f'/bestloss_{varReduction}.txt', test_accuracy)
            np.save(ourdir + f'/valloss_{varReduction}.txt', val_accuracy)
            bestModel = torch.save(model, ourdir + f'/bestmodel_{varReduction}')


    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Main Loss: {main_loss_sim:.4f}',# Adv Loss: {adv_loss_combined:.4f}, '
          #f'Consistency Loss: {consistency_loss:.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}',
          f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
fig, ax = plt.subplots()
#plt.plot(combined, label='Combined Loss')
plt.plot(supervised, label='Task Loss')
plt.xlabel('Epoch No.', fontsize = 18)
plt.ylabel('Loss', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
fig.savefig(f'MLP_losscurve_{varReduction}.pdf')
fig.savefig(f'MLP_losscurve_{varReduction}.png')
plt.show()


# ROC curve and AUC
with torch.no_grad():
    val_real_predictions = torch.sigmoid(test_outputs)  # Apply sigmoid to get probabilities
    test_preds = torch.sigmoid(test_outputs).cpu().numpy() > 0.5
    #test_preds = test_outputs.cpu().numpy() > 0
    test_accuracy = (test_preds.flatten() == y_val.cpu().numpy()).mean()
    print(f'test accuracy = {test_accuracy}')
    fpr, tpr, _ = roc_curve(y_val.cpu().numpy(), val_real_predictions.cpu().numpy())
    roc_auc = auc(fpr, tpr)

    # Save ROC data
    np.save(ourdir + f'/ROC_{varReduction}.npy', np.array([fpr, tpr]))

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'mlp_roc_curve_{varReduction}.png')
    plt.savefig(f'mlp_roc_curve_{varReduction}.pdf')
    plt.show()
