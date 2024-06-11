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
sns.set_theme(font_scale = 1)


varReduction = 3

#Data loading and preprocessing data_path = 'AlephBtag_MC_train_Nev500000.csv'
data_path = 'AlephBtag_MC_train_Nev500000.csv'
data = pd.read_csv(data_path)
print(data.describe())
ourdir = f'true_roc_adv_basic_transform_sin3/{varReduction}'
os.makedirs(ourdir, exist_ok = True)

columns = data.columns
X = data.drop(columns=columns[-1])
y = data[columns[-1]]

#train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#Perturbing the training data to simulate real data
#varReduction = 5  #Change this parameter to change the 'noise' in the real dataset
#X_train_real = 1*(1+np.random.uniform(0,1,X_train.shape)/varReduction)*X_train.copy() + np.random.normal(0, X_train.std(axis=0) / varReduction, X_train.shape)
#X_val_real = 1*(1+np.random.uniform(0,1,X_val.shape)/varReduction)*X_val.copy() + np.random.normal(0, X_val.std(axis=0) / varReduction, X_val.shape)
#X_val = 1*(1+np.random.uniform(0,1,X_val.shape))*X_val.copy() + np.random.normal(0, X_val.std() / varReduction, X_val.shape)



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
        ourNewTensor = ourNewTensor
    varFactor = 25
    ourNewTensor  = ourNewTensor*(1+val) + np.random.normal(0, np.abs(ourNewTensor)/ (varReduction*varFactor), ourNewTensor.shape)
    ourNewTensor = ourNewTensor * ourMaxes
    return ourNewTensor
X_train_real = ourTransform(X_train, 1, varReduction = varReduction)
#X_train = ourTransform(X_train, 1, varReduction = varReduction)
X_val_real = ourTransform(X_val, 1, varReduction = varReduction)
X_val = X_val.copy()#ourTransform(X_val, 4, varReduction = varReduction)
X_train = np.array(X_train)
#X_train_real = 1*(1+np.random.uniform(0,1,X_train.shape)/varReduction)*X_train_real.copy() + np.random.normal(0, X_train_real.std(axis=0) / varReduction, X_train.shape)
#X_val_real = 1*(1+np.random.uniform(0,1,X_val.shape)/varReduction)*X_val_real.copy() + np.random.normal(0, X_val_real.std(axis=0) / varReduction, X_val.shape)
#X_val = 1*(1+np.random.uniform(0,1,X_val.shape))*X_val.copy() + np.random.normal(0, X_val.std() / varReduction, X_val.shape)
import matplotlib.pyplot as plt
#plt.scatter(X_train[::,0], X_train_real[::,0])
#plt.show()
print(X_train.std(axis=0))
print(X_train.mean(axis=0))
print(X_train[0,:])
print(X_train_real[0,:])




#Impute and scale the data
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_train_real_imputed = imputer.transform(X_train_real)
X_val_imputed = imputer.transform(X_val)
X_val_real_imputed = imputer.transform(X_val_real)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_train_real_scaled = scaler.transform(X_train_real_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_val_real_scaled = scaler.transform(X_val_real_imputed)

#Convert to tensors (for torch)
def to_tensor(data, labels=None):
    if labels is not None:
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels.values, dtype=torch.long)
    else:
        return torch.tensor(data, dtype=torch.float32)

X_train_tensor, y_train_tensor = to_tensor(X_train_scaled, y_train)
X_train_real_tensor = to_tensor(X_train_real_scaled)
X_val_tensor, y_val_tensor = to_tensor(X_val_scaled, y_val)
X_val_real_tensor = to_tensor(X_val_real_scaled)

#Move tensors to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_train_real_tensor = torch.tensor(X_train_real_tensor).to(device)
X_val_tensor = X_val_tensor.to(device)
y_val_tensor = y_val_tensor.to(device)
X_val_real_tensor = torch.tensor(X_val_real_tensor).to(device)
#X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
#X_train_real_tensor = X_train_real_tensor.to(device)
#X_val_tensor, y_val_tensor = X_val_tensor.to(device), y_val_tensor.to(device)
#X_val_real_tensor = X_val_real_tensor.to(device)

#Define dimensions of the networks.
input_dim = X_train.shape[1]
hidden_dim = 10
output_dim = 1
adv_hidden_dim = 20
lamb = 50

#Gradient Reversal Layer - simply allows for the training of generator.
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

#Feature Extractor - the generator
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

#Label Predictor - the main task classifier
class LabelPredictor(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(LabelPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.Linear(int(hidden_dim/2), output_dim),
            #nn.Sigmoid()  #Assuming binary classification - i.e. change if using a regressor
        )

    def forward(self, x):
        return self.network(x)

#Domain Discriminator - adversarial component (simulated vs. real)
class DomainDiscriminator(nn.Module):
    def __init__(self, hidden_dim, adv_hidden_dim):
        super(DomainDiscriminator, self).__init__()
        self.network = nn.Sequential(
            #nn.Linear(hidden_dim, adv_hidden_dim),
            #nn.ReLU(),
            nn.Linear(hidden_dim, int(adv_hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(adv_hidden_dim/2), 1),  #Binary domain classification
            #nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

#Simply accuracy function
def compute_accuracy(predictions, labels):
    preds = (predictions > 0.5).float()
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.size(0)
    return accuracy


#Instantiate models and move the models to GPU (necessary)
feature_extractor = FeatureExtractor(input_dim, hidden_dim).to(device)
label_predictor = LabelPredictor(hidden_dim, output_dim).to(device)
domain_discriminator = DomainDiscriminator(hidden_dim, adv_hidden_dim).to(device)
grl = GradientReversalLayer(alpha=1.0).to(device)

#Optimizers
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(label_predictor.parameters()), lr=0.001)
optimizer1 = optim.Adam(list(feature_extractor.parameters()) + list(domain_discriminator.parameters()), lr = 0.001)

#Learning rate scheduler - divides LR by two if there is no loss improvement in 5 epochs.

#Loss functions
classification_loss_fn = nn.BCEWithLogitsLoss()  #Binary Cross Entropy for label prediction
domain_loss_fn = nn.BCEWithLogitsLoss()  #Negative Log Likelihood for domain classification

#Training loop
num_epochs = 1000
batch_size = int(X_train.shape[0]/16)  #Increase batch size for faster training but perhaps more variability.
lr_step = (num_epochs / 10)

ourBestLoss = 0
patience = 100
best_val_loss = np.inf
best_domain_loss = np.inf
counter = 0
scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step, gamma = 0.5)
scheduler1 = optim.lr_scheduler.StepLR(optimizer1, lr_step, gamma = 0.5)

supervised = []
domain = []
combined = []
#permutation = X_train_tensor
for epoch in range(num_epochs):
    feature_extractor.train()
    label_predictor.train()
    domain_discriminator.train()

    permutation = torch.randperm(X_train_tensor.size(0)).cuda()

    epoch_classification_loss = 0.0
    epoch_domain_loss = 0.0

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        sim_data, sim_labels = X_train_tensor[indices], y_train_tensor[indices]
        real_data = X_train_real_tensor[indices]

        #Feature extraction

        sim_features = feature_extractor(sim_data)
        #with torch.no_grad():

        #Supervised loss (label prediction on simulated data)
        predictions = label_predictor(sim_features).squeeze()
        supervised_loss = classification_loss_fn(predictions, sim_labels.float())

        #Domain adaptation loss (adversarial loss on simulated and real data)

        #Total loss
        loss = supervised_loss 
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(feature_extractor.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(label_predictor.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(domain_discriminator.parameters(), max_norm=1.0)
        optimizer.step()

        sim_features = feature_extractor(sim_data)
        real_features = feature_extractor(real_data)
        combined_features = torch.cat((sim_features, real_features), 0)
        domain_labels = torch.cat((torch.zeros(sim_features.size(0)).float().to(device), torch.ones(real_features.size(0)).float().to(device)), 0)
        domain_predictions = domain_discriminator(grl(combined_features)).squeeze()
        domain_loss = domain_loss_fn(domain_predictions, domain_labels)
        loss1 = domain_loss*lamb


        #Backpropagation with gradient clipping
        optimizer1.zero_grad()
        loss1.backward()
        nn.utils.clip_grad_norm_(feature_extractor.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(label_predictor.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(domain_discriminator.parameters(), max_norm=1.0)
        optimizer1.step()

        epoch_classification_loss += supervised_loss.item()
        epoch_domain_loss += domain_loss.item()

    epoch_classification_loss /= (X_train_tensor.size(0) // batch_size)
    epoch_domain_loss /= (X_train_tensor.size(0) // batch_size)
    supervised.append(epoch_classification_loss)
    domain.append(epoch_domain_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Classification Loss: {epoch_classification_loss:.4f}, Domain Loss: {epoch_domain_loss:.4f}')

    #Evaluation
    feature_extractor.eval()
    label_predictor.eval()
    domain_discriminator.eval()

    with torch.no_grad():
        #Evaluate on simulated validation data
        val_predictions = label_predictor(feature_extractor(X_val_tensor)).squeeze()
        val_loss = classification_loss_fn(val_predictions, y_val_tensor.float())
        val_accuracy = compute_accuracy(val_predictions, y_val_tensor.float())

        #Evaluate on real validation data
        val_real_predictions = label_predictor(feature_extractor(X_val_real_tensor)).squeeze()
        val_real_loss = classification_loss_fn(val_real_predictions, y_val_tensor.float())
        val_real_accuracy = compute_accuracy(val_real_predictions, y_val_tensor.float())

        print(f'Validation Loss: {val_loss.item():.4f}')
        print(f'Validation Loss on Real Data: {val_real_loss.item():.4f}')
        print(f'Validation Accuracy: {val_accuracy.item():.4f}')
        print(f'Validation Accuracy on Real Data: {val_real_accuracy.item():.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr(), scheduler1.get_last_lr()}')
        if val_real_accuracy.item() > ourBestLoss:
            ourBestLoss = val_real_accuracy.item()
            val_best_predictions = val_real_predictions
            np.save(ourdir + f'/bestloss_{varReduction}.txt', val_real_accuracy.item())
            np.save(ourdir + f'/valloss_{varReduction}.txt', val_accuracy.item())
            torch.save(feature_extractor, ourdir + f'/feature_extractor_{varReduction}')
            torch.save(label_predictor, ourdir + f'/label_predictor_{varReduction}')
            torch.save(domain_discriminator, ourdir + f'/domain_discriminator_{varReduction}')

        #Learning rate adjustment
        scheduler.step()
        scheduler1.step()
        if scheduler.get_last_lr()[0] < 10e-5:
            break
    if epoch_domain_loss < best_domain_loss:
        best_domain_loss = epoch_domain_loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        #torch.save(model.state_dict(), 'best_model.pt')
    else:
        counter += 1
        if counter >= patience and epoch_domain_loss > best_domain_loss:
            print(f'Early stopping at epoch {epoch + 1}')
            break
        else:
            counter = 0

#Evaluation
feature_extractor.eval()
label_predictor.eval()
domain_discriminator.eval()

with torch.no_grad():
    #Evaluate on simulated validation data
    val_predictions = label_predictor(feature_extractor(X_val_tensor)).squeeze()
    val_loss = classification_loss_fn(val_predictions, y_val_tensor.float())
    val_accuracy = compute_accuracy(val_predictions, y_val_tensor.float())

    #Evaluate on real validation data
    val_real_predictions = label_predictor(feature_extractor(X_val_real_tensor)).squeeze()
    val_real_loss = classification_loss_fn(val_real_predictions, y_val_tensor.float())
    val_real_accuracy = compute_accuracy(val_real_predictions, y_val_tensor.float())

    print(f'Validation Loss: {val_loss.item():.4f}')
    print(f'Validation Loss on Real Data: {val_real_loss.item():.4f}')
    print(f'Validation Accuracy: {val_accuracy.item():.4f}')
    print(f'Validation Accuracy on Real Data: {val_real_accuracy.item():.4f}')

fig, ax = plt.subplots()
#plt.plot(combined, label='Combined Loss')
plt.plot(domain, label='Domain Loss')
plt.plot(supervised, label='Task Loss')
plt.xlabel('Epoch No.', fontsize = 18)
plt.ylabel('Loss', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend()
plt.tight_layout()
fig.savefig(ourdir + f'/DANN_losscurve_{varReduction}.pdf')
fig.savefig(ourdir + f'/DANN_losscurve_{varReduction}.png')
plt.show()

with torch.no_grad():
    val_real_predictions = torch.sigmoid(val_best_predictions)  # Apply sigmoid to get probabilities
    fpr, tpr, _ = roc_curve(y_val_tensor.cpu().numpy(), val_real_predictions.cpu().numpy())
    roc_auc = auc(fpr, tpr)

    # Save ROC data
    np.save(ourdir + '/ROC.npy', np.array([fpr, tpr]))

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
    plt.savefig(ourdir + f'/DANN_roc_curve_{varReduction}.png')
    plt.savefig(ourdir + f'/DANN_roc_curve_{varReduction}.pdf')
    plt.show()
