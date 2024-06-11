import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

A = 6  # Want figures to be A6
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 35.61 * .5**(.5 * A)])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 36})
sns.set(font_scale = 1)

dir1 = "roc_mlp_transform_sin3/"
MLPLoss = np.zeros(6)
MLPfpr = np.zeros(6, dtype=object)
MLPtpr = np.zeros(6, dtype=object)
MLPAUC = np.zeros(6)
for root, subdirs, files in os.walk(dir1):
    if files != []:
        files = list(files)
        for filename in files:
            if filename[0:8] == 'bestloss':
                index = int(filename[9])-1
                if index >= len(MLPLoss):
                    index = len(MLPLoss)-1
                x = np.load(root + '/' + filename)
                MLPLoss[index] = x
            if filename[0:3] == 'ROC':
                print(filename[5])
                index = int(filename[4])-1
                if index >= len(MLPLoss):
                    index = len(MLPLoss)-1
                MLPfpr[index], MLPtpr[index] = np.load(root + '/' + filename)
                MLPAUC[index] = auc(MLPfpr[index], MLPtpr[index])


dir2 = "roc_adv_one_transform_sin3/"
AdvOneLoss = np.zeros(6)
AdvOnefpr = np.zeros(6, dtype=object)
AdvOnetpr = np.zeros(6, dtype=object)
AdvOneAUC = np.zeros(6)
for root, subdirs, files in os.walk(dir2):
    if files != []:
        files = list(files)
        for filename in files:
            if filename[0:8] == 'bestloss':
                index = int(filename[9])-1
                if index >= len(AdvOneLoss):
                    index = len(AdvOneLoss)-1
                print(index)
                x = np.load(root + '/' + filename)
                AdvOneLoss[index] = x
            if filename[0:3] == 'ROC':
                print(filename[5])
                index = int(filename[4])-1
                if index >= len(AdvOneLoss):
                    index = len(AdvOneLoss)-1
                AdvOnefpr[index], AdvOnetpr[index] = np.load(root + '/' + filename)
                AdvOneAUC[index] = auc(AdvOnefpr[index], AdvOnetpr[index])




dir3 = "roc_adv_basic_transform_sin3/"
AdvBasicLoss = np.zeros(6)
AdvBasicfpr = np.zeros(6, dtype=object)
AdvBasictpr = np.zeros(6, dtype=object)
AdvBasicAUC = np.zeros(6)
for root, subdirs, files in os.walk(dir3):
    if files != []:
        files = list(files)
        for filename in files:
            if filename[0:8] == 'bestloss':
                index = int(filename[9])-1
                if index >= len(AdvBasicLoss):
                    index = len(AdvBasicLoss)-1
                print(index)
                x = np.load(root + '/' + filename)
                AdvBasicLoss[index] = x
            if filename[0:3] == 'ROC':
                print(filename[5])
                index = int(filename[4])-1
                if index >= len(AdvBasicLoss):
                    index = len(AdvBasicLoss)-1
                AdvBasicfpr[index], AdvBasictpr[index] = np.load(root + '/' + filename)
                AdvBasicAUC[index] = auc(AdvBasicfpr[index], AdvBasictpr[index])


fig, ax = plt.subplots()
x = np.array([0,1,2,3,4,7])+1
sns.lineplot(x = x, y = MLPLoss, label = "MLP", color = 'blue')
sns.lineplot(x = x, y = AdvBasicLoss, label = "DANN", color = 'green')
sns.lineplot(x = x, y = AdvOneLoss, label = "Hybrid-DANN", color = 'orange')
plt.xlabel('recipricol noise, $t$', fontsize = 18)
plt.ylabel('real data accuracy, $\\alpha$', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend()
plt.tight_layout()
fig.savefig('accuracy_sin3.pdf')
plt.show()


fig, ax = plt.subplots(2,2)
#ax[0][0].scatter(x, MLPAUC, label = "MLP")
#ax[0][0].scatter(x, AdvBasicAUC, label = "DANN")
#ax[0][0].scatter(x, AdvOneAUC, label = "Hybrid-DANN")
#ax[0][0].set_xlabel('recipricol noise, $t$')#, fontsize = 18)
#ax[0][0].set_ylabel('ROC')#, fontsize = 18)


ax[0][0].scatter(MLPfpr[0], MLPtpr[0], lw=2, color = 'blue', s = 0.25)
ax[0][0].scatter(AdvBasicfpr[0], AdvBasictpr[0], lw=2, color = 'green', s = 0.25)
ax[0][0].scatter(AdvOnefpr[0], AdvOnetpr[0], lw=2, color = 'orange', s = 0.25)
ax[0][0].set_xlim([0.0, 1.0])
ax[0][0].set_ylim([0.0, 1.05])
ax[0][0].scatter([0, 1], [0, 1], color='navy', lw=2, linestyle='--', s = 0.25)
ax[0][0].set_xlabel('False Positive Rate')
ax[0][0].set_ylabel('True Positive Rate')
ax[0][0].set_title('Noise level $t = 1$')


ax[0][1].scatter(MLPfpr[2], MLPtpr[2], lw=2, color = 'blue', s = 0.25)
ax[0][1].scatter(AdvBasicfpr[2], AdvBasictpr[2], lw=2, color = 'green', s = 0.25)
ax[0][1].scatter(AdvOnefpr[2], AdvOnetpr[2], lw=2, color = 'orange', s = 0.25)
ax[0][1].set_xlim([0.0, 1.0])
ax[0][1].set_ylim([0.0, 1.05])
ax[0][1].scatter([0, 1], [0, 1], color='navy', lw=2, linestyle='--', s = 0.25)
ax[0][1].set_xlabel('False Positive Rate')
ax[0][1].set_ylabel('True Positive Rate')
ax[0][1].set_title('Noise level $t = 3$')


ax[1][0].scatter(MLPfpr[4], MLPtpr[4], lw=2, color = 'blue', s = 0.25)
ax[1][0].scatter(AdvBasicfpr[4], AdvBasictpr[4], lw=2, color = 'green', s = 0.25)
ax[1][0].scatter(AdvOnefpr[4], AdvOnetpr[4], lw=2, color = 'orange', s = 0.25)
ax[1][0].set_xlim([0.0, 1.0])
ax[1][0].set_ylim([0.0, 1.05])
ax[1][0].scatter([0, 1], [0, 1], color='navy', lw=2, linestyle='--', s = 0.25)
ax[1][0].set_xlabel('False Positive Rate')
ax[1][0].set_ylabel('True Positive Rate')
ax[1][0].set_title('Noise level $t = 5$')


#ax[1][1].scatter(MLPfpr[5], MLPtpr[5], lw=2)
#ax[1][1].scatter(AdvBasicfpr[5], AdvBasictpr[5], lw=2)
#ax[1][1].scatter(AdvOnefpr[5], AdvOnetpr[5], lw=2)
#ax[1][1].set_xlim([0.0, 1.0])
#ax[1][1].set_ylim([0.0, 1.05])
#ax[1][1].scatter([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#ax[1][1].set_xlabel('False Positive Rate')
#ax[1][1].set_ylabel('True Positive Rate')
#ax[1][1].set_title('Noise level $t = 8$')
ax[1][1].scatter(x, MLPAUC, label = "MLP", color = 'blue')
ax[1][1].scatter(x, AdvBasicAUC, label = "DANN", color = 'green')
ax[1][1].scatter(x, AdvOneAUC, label = "Hybrid-DANN", color = 'orange')
ax[1][1].set_xlabel('recipricol noise, $t$')#, fontsize = 18)
ax[1][1].set_ylabel('ROC')#, fontsize = 18)
plt.legend()


plt.tight_layout()
fig.savefig('ROC_sin3.pdf')
fig.savefig('ROC_sin3.png')
plt.show()
