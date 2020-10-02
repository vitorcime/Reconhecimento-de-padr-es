import json
import mne
import scipy
import matplotlib
import numpy as np
from sklearn.svm import SVC
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from google_drive_downloader import GoogleDriveDownloader as gdd


dataset_ids = {
    1: '1ym3JqI4_ZYNSBLxoF1vNxI5Hsgur_tha',
    2: '1tJ5tHbE-2jwr0gA33Gd873lRPao-e4dF',
    3: '1tXdpY-mser01POaP6Qwixny6LjsXXoXB',
    4: '1T00cliWp5yqqbbWZ7-rf2X4tOUQ-PvIQ',
    5: '1CYsBFMul9zB_vCy0XD7XVfKUl8vihDYZ',
    6: '1io6jdALBKeopELWXahPzuAi6EfYDgviW',
    7: '1YDkheRDPNDR1ujsqqC_SY6cebWHkw9Xt',
    8: '1jjoQJFDCi7O9Q-iaReAPpQnxC-HIKpQi',
}
label_id = '1mD5MXoh6tfQJFXIvdw2MQsEu6vZka6C0'
desc = '14kYNBZYdttqmSS_Vz6Bm_ztG9Uw1MC0y'

# ALTERE O ID DO DATASET DE SUA ESCOLHA AQUI ##################################
DS = 4


# In[3]:


# download do stataset
gdd.download_file_from_google_drive(file_id=dataset_ids[DS],
                                    dest_path='files/data.npy',
                                    showsize=True)
# download do arquivo de marcações
gdd.download_file_from_google_drive(file_id=label_id,
                                    dest_path='files/labels.npy', showsize=True)

# download do arquivo de descrição
gdd.download_file_from_google_drive(file_id=desc,
                                    dest_path='files/descriptor.json',
                                    showsize=True)

data = np.load('files/data.npy')
labels = np.load('files/labels.npy')

desc_file = open('files/descriptor.json')
deor = json.loads(desc_file.read())
desc_file.close()

print('Estruturas => dados', data.shape, 'labels', labels.shape)
print(labels)

data = data[:,:256,:]

trial_duration = 5
sampling_frequency = data.shape[-1] / trial_duration
montage = mne.channels.make_standard_montage('EGI_256')
ch_names = data.shape[1]
ch_types = 'eeg'


info = mne.create_info(montage.ch_names, sampling_frequency, ch_types)


info.set_montage(montage)


events = np.array([[index, 0, event] for index, event in enumerate(labels)])

epoch = mne.EpochsArray(data, info, events)

filtered_epoch = epoch.copy().pick_channels(['E108', 'E109', 'E116', 'E125', 'E118', 'E117', 'E126',
                      'E139', 'E127', 'E138', 'E140', 'E150', 'E151'])
filtered_epoch.filter(l_freq = 5.0, h_freq = 14.0)


X, _ = mne.time_frequency.psd_multitaper(filtered_epoch, fmin=5.0, fmax=14.0)



X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])


y = np.load('files/labels.npy')


for count in range(50):
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        for gamma in [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
            for C in [0.01, 0.1, 1, 10, 100, 1000]:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
                clf = SVC(gamma=gamma, kernel=kernel, C=C)
                clf.fit(X_train, y_train)
                res = clf.predict(X_test)
                tot_hit = sum([1 for i in range(len(res)) if res[i] == y_test[i]])
                if tot_hit / X_test.shape[0] * 100 > 50:
                    print('Acurácia: {:.2f}%'.format(tot_hit / X_test.shape[0] * 100))

