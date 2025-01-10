# Replace the import statement for ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Rest of your code remains unchanged
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # serve a disattivare la GPU (non viene vista), perciò occorre commentare/decommentare questa riga per attivarla/disattivarla
import numpy as np
seed = 2018
np.random.seed(seed)
import random
import time

import librosa
from scipy import signal

import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Dense
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras import Model

from keras.utils import to_categorical

# Use keras.layers instead of keras.layers.merge
from keras.layers import Concatenate

from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50
from skimage.transform import resize
import matplotlib.pyplot as plt







##############################################################

current_model = DenseNet121
#current_model = ResNet50

model_name = 'TreeVibes_' + current_model.__name__

log_path = model_name + '.log'
batch_size = 4 # it was originally 16
epochs = 100
es_patience = 7
rlr_patience = 3

SR = 8000
N_FFT = 256
HOP_LEN = int(N_FFT / 2)
input_shape = (129, 1251, 1)



##############################################################

target_names = ['clean', 'infested'] # label delle classi
X_names = []
y = []
target_count = []

for i, target in enumerate(target_names):
    target_count.append(0)
    path = 'treevibes/field/field/train/' + target + '/' # train from scratch using field data
    for [root, dirs, files] in os.walk(path): # per tutto ciò che c'è annidato all'interno della cartella indicata da path
        for filename in files: # per tutti i files
            name,ext = os.path.splitext(filename) # prelevo nome ed estensione del file corrente
            if ext == '.wav': # guardo solo i file .wav
                name = os.path.join(root, filename) # nome del file a partire dalla root che è 'treevibes', e.g.: 'treevibes/field/field/train/clean/folder_25/F_20200524145254_2.wav'
                y.append(i) # assegnazione del label '0' ai file di 'clean', e del label '1' ai file di 'infested'
                X_names.append(name) # array dei files controllati
                target_count[i]+=1
    print (target, '#recs = ', target_count[i])

print ('total #recs = ', len(y))

X_names, y = shuffle(X_names, y, random_state = seed) # i nomi dei path vengono messi in ordine sparso così da poter usarne i file nel passaggio successivo dividendoli in training e testing
X_train, X_test, y_train, y_test = train_test_split(X_names, y, stratify = y, test_size = 0.20, random_state = seed) # i file verranno suddivisi 80% training e 20% testing

print ('train #recs = ', len(X_train)) # 80% dei file per training
print ('test #recs = ', len(X_test)) # 20% dei file per testing
##############################################################################






# *********************************************************************
# ROBA MIA
#defining the display function
def GFG(arr,prec):
    np.set_printoptions(suppress=True,precision=prec)
    print(arr)

train_len = 12
batch_dim = 4
for start in range(0, train_len, 4):
    x_batch_dummy = []
    y_batch_dummy = []

    dummy_matrix = np.zeros((200,200))

    for i in range (200):
        for j in range(200):
            dummy_matrix[i][j] = i*200+j

    print("\ndummy_matrix before:\n")
    GFG(dummy_matrix, 0)
    print("\ndummy_matrix after:\n")
    dummy_matrix = dummy_matrix[:, :15]
    GFG(dummy_matrix, 0)
    dummy_matrix = np.flipud(dummy_matrix)
    print("\ndummy_matrix flipped rows:\n")
    GFG(dummy_matrix, 0)
    dummy_matrix = np.expand_dims(dummy_matrix, axis=-1)
    print("\ndummy_matrix after exp dim:\n")
    GFG(dummy_matrix, 0)


    print('len(X_test):', len(X_test))

# *********************************************************************








def train_generator():
    print("__ enter train_generator:")
    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end] # prelevo i batch di file da allenare...
            labels_batch = y_train[start:end] # ... e i loro label

            '''
            print("\n\n******************************************************")
            print('len(X_train): ', len(X_train))
            print('len(train_batch): ', len(train_batch))
            print("******************************************************\n\n")
            '''

            for i in range(len(train_batch)):
                data, rate = librosa.load(train_batch[i], sr=SR) # carico i file audio come una serie di floating point, e il rate sarà pari al sample rate (SR), in questo caso 8KHz -> 8000
                data = np.roll(data, random.randint(0, len(data))) # ???

                if len(data) < 20 * rate: data = np.repeat(data, int(20 * rate / len(data) + 1)) # ???

                data = librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LEN) # applicazione della short time Furier transform, con  frame di N_FFT=256 samples, e una distanza tra un frame e l'altro pari a HOP_LEN = N_FFT/2 = 128 samples
                data = librosa.amplitude_to_db(abs(data)) # trasforazione dell'ampiezza in dB
                data = data[:, :1251] # vengono prelevati i primi 1251 elementi di ogni riga della matrice 'data' >>> ottengo una matrice con #RIGHE = #righe e #COLONNE = 1251

                data = np.flipud(data) # viene invertito l'ordine delle righe
                # data = resize(data,(129,1200))

                data = np.expand_dims(data, axis=-1) # viene aggiunta una dimensione, che consiste nel riporre ogni numero di ogni riga di array, in un array a se (cioè ogni riga avrà 1251 array, ognuno contenente un numero)

                x_batch.append(data) # viene accodato l'attuale 'data' nel batch di training da costruire
                y_batch.append(labels_batch[i]) # viene associato un label all'attuale 'data' appena accodato al batch di training da costruire

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            y_batch = to_categorical(y_batch, len(target_names))
            print("__ exit train_generator:")
            yield x_batch, y_batch


def valid_generator():
    print("__ enter valid_generator:")
    while True:
        for start in range(0, len(X_test), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_test))
            test_batch = X_test[start:end]
            labels_batch = y_test[start:end]

            '''
            print("\n\n******************************************************")
            print('len(X_test): ', len(X_test))
            print('len(test_batch): ', len(test_batch))
            print("******************************************************\n\n")
            '''

            for i in range(len(test_batch)):
                data, rate = librosa.load(test_batch[i], sr=SR)
                data = np.roll(data, random.randint(0, len(data)))

                if len(data) < 20 * rate: data = np.repeat(data, int(20 * rate / len(data) + 1))

                data = librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LEN)
                data = librosa.amplitude_to_db(abs(data))
                data = data[:, :1251]

                data = np.flipud(data)
                # data = resize(data,(129,1200))

                data = np.expand_dims(data, axis=-1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            y_batch = to_categorical(y_batch, len(target_names))
            print("__ exit valid_generator:")
            yield x_batch, y_batch


def test_generator():
    print("__ enter test_generator:")
    while True:
        for start in range(0, len(X_test), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_test))
            test_batch = X_test[start:end]
            labels_batch = y_test[start:end]

            '''
            print("\n\n******************************************************")
            print('len(X_test): ', len(X_test))
            print('len(test_batch): ', len(test_batch))
            print("******************************************************\n\n")
            '''

            for i in range(len(test_batch)):
                data, rate = librosa.load(test_batch[i], sr=SR)

                if len(data) < 20 * rate: data = np.repeat(data, int(20 * rate / len(data) + 1))

                data = librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LEN)
                data = librosa.amplitude_to_db(abs(data))
                data = data[:, :1251]

                data = np.flipud(data)
                # data = resize(data,(129,1200))

                data = np.expand_dims(data, axis=-1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            y_batch = to_categorical(y_batch, len(target_names))
            print("__ exit test_generator:")
            yield x_batch, y_batch

##############################################################################



from keras.callbacks import ModelCheckpoint
import tensorflow as tf #aggiunto io *********************************************
import math
# Rest of your code remains unchanged
img_input = Input(shape=input_shape)

model = current_model(input_tensor=img_input, classes=len(target_names), weights=None) # all'inizio di questo codice, è stato scritto: current_model = DenseNet121.
                                                                                        # weights=None significa che i pesi verranno generati durante il nostro training,
                                                                                        # cioè non verranno usati quelli di pre-training (pesi standard di Densenet121).
# model = ResNet50(input_tensor=img_input, classes=len(target_names), weights=None)

# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"]) # original

#cliccare su tf.keras.metrics per vedere tutte le metriche disponibili! ********************************************************************************************************
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy", tf.keras.metrics.Recall(),tf.keras.metrics.Precision()]) #aggiunto io *********************************************

# Specify the correct filepath for saving the whole model (architecture + weights)
ckpt = ModelCheckpoint("TreeVibe_model.keras", save_best_only=True)  # Use .keras extension

es = EarlyStopping("val_categorical_accuracy", restore_best_weights=True, patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

print("BEFORE MODEL.FIT")
hist = model.fit(
    train_generator(),
    steps_per_epoch=int(math.ceil(float(len(X_train)) / float(batch_size))),
    validation_data=valid_generator(),
    validation_steps=int(math.ceil(float(len(X_test)) / float(batch_size))),
    epochs=10,
    callbacks=[ckpt, es, reduce_lr],
    shuffle=False
)
print("AFTER MODEL.FIT")

# Save the whole model (architecture + weights)
model.save('TreeVibe_model.keras')
model.save('TreeVibe_model.h5')

time.sleep(3)


#load h5 module
model_h5=tf.keras.models.load_model('/home/zord/PycharmProjects/TF_test/RPW_tf/TreeVibe_model.h5')
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model_h5)

time.sleep(3)

#convert to tflite
tflite_model = tflite_converter.convert()
open("TreeVibe_model.tflite", "wb").write(tflite_model)

time.sleep(3)

interpreter = tf.lite.Interpreter(model_path='/home/zord/PycharmProjects/TF_test/RPW_tf/TreeVibe_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)


# Print the whole model (weigths and biases for every layer)
#print(model.variables)

# Saving the whole model weights in numpy format. This will create a directory named ‘weights’ with subdirectories for each model layer.
from pathlib import Path
for variable in model.variables:
    path_list = variable.name.replace(":0","").split("/")
    dir_path = "weights/"+ ("/").join(path_list[:-1])
    file_name = path_list[-1]
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    np.save(dir_path+"/"+file_name+".npy",variable.numpy())

# Load the model later using:ZZ```````
# loaded_model = keras.models.load_model('TreeVibe_model.keras')




import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import librosa
# import numpy as np #***********************************
from keras.models import load_model

# Load the saved model
model = load_model('TreeVibe_model.keras')

# Function to preprocess audio files
def preprocess_audio(audio_path, sr=8000, n_fft=256, hop_length=128):
    print("__ enter def preprocess_audio(audio_path, sr=8000, n_fft=256, hop_length=128):")
    data, _ = librosa.load(audio_path, sr=sr) # gli audio contenuti nella cartella specificata 'audio_path' vengono caricati all'interno di 'data', e poi recupera il sample rate 'sr'
    # Perform additional preprocessing if needed
    # Example: Compute spectrogram, reshape, etc.
    data = np.roll(data, np.random.randint(0, len(data)))  # Random roll as in training
    if len(data) < 20 * sr:
        data = np.repeat(data, int(20 * sr / len(data) + 1)) # se non ci sono abbastanza dati (<20*sr), allora li replico; e.g. x = np.array([[1,2],[3,4]]) se faccio np.repeat(x, 3) ottengo: array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    data = librosa.stft(data, n_fft=n_fft, hop_length=hop_length) # short time Fourier transform, con frame di 256 samples, e salto tra un frame e l'altro di 128 samples
    data = librosa.amplitude_to_db(abs(data)) # da amplitude a dB
    #print("\n\n=============================DATA=================================")
    #print(data)
    #print("=============================DATA=================================\n\n")
    data = data[:, :1251] # vengono prelevati i primi 1251 elementi di ogni riga della matrice 'data' >>> ottengo una matrice con #RIGHE = #righe e #COLONNE = 1251
    data = np.flipud(data) # viene invertito l'ordine delle righe
    data = np.expand_dims(data, axis=-1) # viene aggiunta una dimensione, che consiste nel riporre ogni numero di ogni riga di array, in un array a se (cioè ogni riga avrà 1251 array, ognuno contenente un numero)
    print("__ exit def preprocess_audio(audio_path, sr=8000, n_fft=256, hop_length=128):")
    return data

# Function to perform inference on audio files
def classify_audio(audio_path):
    print("__ enter def classify_audio(audio_path):")
    # Preprocess the audio
    processed_audio = preprocess_audio(audio_path)
    # Perform inference using the loaded model
    prediction = model.predict(np.expand_dims(processed_audio, axis=0))
    # Interpret the prediction
    class_idx = np.argmax(prediction)
    print("__ exit def classify_audio(audio_path):")
    return class_idx

# Function to classify all audio files in a folder
def classify_folder(folder_path):
    print("__ enter def classify_folder(folder_path):")
    audio_files = []
    predicted_classes = []
    target_names = ['clean', 'infested']  # Define your target class names

    # Classify each audio file in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
                predicted_classes.append(classify_audio(os.path.join(root, file)))
    print("__ exit def classify_folder(folder_path):")
    return audio_files, predicted_classes

# Define folder paths
clean_folder = 'treevibes/field/field/train/clean/folder_25'
infested_folder = 'treevibes/field/field/train/infested/folder_19'

# Classify audio files in the clean and infested folders
clean_files, clean_predicted = classify_folder(clean_folder)
infested_files, infested_predicted = classify_folder(infested_folder)

# Concatenate predicted classes and ground truth labels
predicted = np.concatenate([clean_predicted, infested_predicted])
true_labels = np.concatenate([np.zeros(len(clean_predicted)), np.ones(len(infested_predicted))])

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted)

# Normalize confusion matrix to get percentage values
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix with percentage values
plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names)
plt.yticks(tick_marks, target_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')


plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()
plt.title('Precision')
plt.plot(hist.history['precision'], 'r')
plt.plot(hist.history['val_precision'], 'b')
plt.show()
plt.title('Recall')
plt.plot(hist.history['recall'], 'r')
plt.plot(hist.history['val_recall'], 'b')
plt.show()


# Add percentage values to the plot
for i in range(conf_matrix_norm.shape[0]):
    for j in range(conf_matrix_norm.shape[1]):
        plt.text(j, i, "{:.2f}%".format(conf_matrix_norm[i, j]),
                 horizontalalignment="center",
                 color="white" if conf_matrix_norm[i, j] > (conf_matrix_norm.max() / 2) else "black")

plt.show()





# Example usage:
audio_file_path = 'treevibes/field/field/train/clean/folder_25/F_20200524144643_1.wav'  # Replace with your audio file path
predicted_class_idx = classify_audio(audio_file_path)
target_names = ['clean', 'infested']  # Define your target class names
predicted_class_name = target_names[predicted_class_idx]
print(f"The audio file '{audio_file_path}' is classified as '{predicted_class_name}'.")