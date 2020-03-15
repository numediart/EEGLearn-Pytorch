import scipy.io as sio
import matplotlib.pyplot as plt

import os
import glob

import numpy as np

Patient_id = np.unique(sio.loadmat("Sample Data/trials_subNums.mat")['subjectNum'][0]) #corresponding to the patient id
dir = "Results/"
models = ["Basic", "LSTM", "MaxCNN", "Mix", "TempCNN"]

Results = np.zeros((len(Patient_id), len(models),20))

inc = 0
for model in models:
    inc_patient = 0
    for patient in Patient_id:
        doc = glob.glob(dir+"*"+model+"*"+"t"+str(patient)+".mat")[0]
        file = sio.loadmat(doc)['res']
        Results[inc_patient,inc, :] = file[:,3]
        inc_patient += 1
    inc += 1

fig = plt.figure()

for i in range(len(models)):
    a = 5
    plt.plot(np.max(Results[:,i,:],axis=1), '.-', label = models[i])

plt.legend()
#plt.boxplot(Results[:,0,:])
#plt.boxplot(Results[:,1,:])



lstm = sio.loadmat("Results/result_LSTM.mat")['vacc']
plt.plot(np.mean(lstm, axis=0))


plt.show()
