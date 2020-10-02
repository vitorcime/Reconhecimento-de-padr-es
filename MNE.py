#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
import json
import mne
import matplotlib
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


# In[2]:


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


# In[4]:


# carregamento
X = np.load('files/data.npy')
y = np.load('files/labels.npy')
desc_file = open('files/descriptor.json')
descriptor = json.loads(desc_file.read())
desc_file.close()
print('Estruturas => dados', X.shape, 'labels', y.shape)


# In[5]:


print('Características do voluntário:', descriptor[str(DS)])
print('\nRótulos:', descriptor['frequencies'])
print('\nTaxa de amostragem:', descriptor['sampling_rate'])


# In[6]:


'''
#criacao mne
X=X[:,:256,:]
ch_names = X.shape[1]
sfreq = X.shape[-1]/5
ch_types = 'eeg'
info = mne.create_info(ch_names, sfreq, ch_types)

'''


# In[7]:


descriptor['sampling_rate'] = X.shape[-1] / 5
print('Nova taxa de amostragem: {} Hz'.format(descriptor['sampling_rate']))


# In[8]:




get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [12, 8]

montage = mne.channels.make_standard_montage('EGI_256')
info = mne.create_info(montage.ch_names,
                       sfreq=descriptor['sampling_rate'],
                       ch_types='eeg')
info.set_montage(montage)


# In[9]:


# o 257º eletrodo é o VREF (referência). Inútil -> Podemos tirá-lo...
X = X[:,:256,:]
# objeto event é uma matriz tridimensional conforme explicado em aula
events = np.array([[i, 0, e] for i, e in enumerate(y)])
# instanciando objeto EpochArray
epoch = mne.EpochsArray(X, info, events=events)


# In[10]:


epoch_ex = epoch.copy().pick_channels(['E108', 'E109', 'E116', 'E125', 'E118', 'E117', 'E126',
                      'E139', 'E127', 'E138', 'E140', 'E150', 'E151'])
epoch_ex.filter(l_freq = 5.0, h_freq = 14.0)
print(epoch.get_data().shape)
print(epoch_ex.get_data().shape)


# In[11]:


matplotlib.rcParams['figure.figsize'] = [6., 4.]

for y in range(1,6):
    for i in (3, 9, 11):
        print("Evento " + str(y))
        print("Rótulos", descriptor['frequencies'])
        epoch_ex[str(y)][-i].plot_psd(fmin = 5., fmax = 14.)
        print()
        

# CAR: Técnica usada para o calculo de uma media dos eletrodos atraves do eletrodo de referencia
# In[12]:


### FIltro

epo_b2 = epoch_ex.copy().filter(l_freq=5, h_freq=None)
epo_b2.filter(l_freq=None, h_freq=14)

a = epoch_ex.get_data()
a = a.transpose(1, 0, 2)
a = a.reshape(13, 125 * 1205)

# criando o objeto `info` (o restante dos valores já temos)
info = mne.create_info(ch_names=13,
                       sfreq=241.,
                       ch_types='eeg')

raw = mne.io.RawArray(a, info)
epo_ref = mne.set_eeg_reference(epoch_ex, ref_channels=['E116', 'E126', 'E150'])


# In[13]:


### Features
data = epoch_ex.get_data()
print(data.shape) # domínio do tempo

# aplicando STFT
_, _, w = stft(data, fs=241, nperseg=32, noverlap=16)
# w = np.swapaxes(w, 3, 4)
print(w.shape)

W = np.abs(w) ** 2

fmn = np.mean(W, axis=-1)
print('FMN:', fmn.shape)

# Root of sum of squares
rss = np.sqrt(np.sum(W, axis=-1))
print('RSS:', rss.shape)
features = list()
for feature in (fmn, rss,):
    feature = feature.transpose(0, 2, 1)
    feature = feature.reshape(feature.shape[0] * feature.shape[1],
                              feature.shape[2])
    features.append(feature)

# vetor de características final
X = np.concatenate(features, axis=-1)
print('Shape dos dados:', X.shape)
y = np.load('files/labels.npy')
print('Shape original dos labels', y.shape)

size = int(X.shape[0] / y.shape[0])
y = np.concatenate([y for i in range(size)])
print('Shape final dos labels', y.shape)


# In[ ]:


###Classificador
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, shuffle=True)
for count in range(50):
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        for gamma in [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
            for C in [0.01, 0.1, 1, 10, 100, 1000]:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=0.7, shuffle=True)
                clf = SVC(kernel=kernel,gamma=gamma, C=C)
                clf = clf.fit(X_train, y_train)
                res = clf.predict(X_test)
                if ( 100*metrics.accuracy_score(y_test, res) >= 10.0): # 92
                    print(count)
                    print('Kernel:{} | Gamma:{} e C:{} | Accuracy: {:.2f}%'.format(
                        kernel, gamma, C, 100*metrics.accuracy_score(y_test, res))
                    )

