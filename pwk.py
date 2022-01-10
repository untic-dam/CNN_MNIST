import matplotlib.pyplot as plt
import pandas as pd
import random

import os, time, sys
import h5py
import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard



def get_callbacks(run_dir):
    #create dir to strore models and logs
    mkdir(run_dir + '/models')
    mkdir(run_dir + '/logs')

    now = tag_now()

    # ---- Callback tensorboard
    log_dir = run_dir + "/logs/tb_" + now
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # ---- Callback ModelCheckpoint - Save best model
    save_dir = run_dir + "/models/m_"+ now +"/best-model.h5"
    bestmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir, verbose=0, monitor='accuracy', save_best_only=True)

    # ---- Callback ModelCheckpoint - Save model each epochs
    save_dir = run_dir + "/models/m_"+ now +"/model-{epoch:04d}.h5"
    savemodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir, verbose=0)

    # ---- Create a dict with all calbacks
    clbcks = {
        'tensorboard' : tensorboard_callback,
        'bestmodel'   : bestmodel_callback,
        'savemodel'   : savemodel_callback
    }

    # --- Print command line to monitor the learning process with Tensorboard
    path=os.path.abspath(f'{run_dir}/logs')
    print('\ncallbacks stored into : ',now)
    print(f'tensorboard --logdir {path} --load_fast true')

    return clbcks


def get_best_model(x_test, y_test, run_dir, show=True):
    #return the best model located into 'run_dir' as a dictionnary 
    #best = {'model' | 'dir' | 'score'}

    #parameters to select the best model
    best = {}
    acc_best = 0.0

    #init folder models
    models_dir = run_dir + '/models'
    sub_folders = [name for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name))]

    #loop to select the best model
    for dir_name in sub_folders:
        pth_best_models = models_dir + '/' + dir_name + '/best-model.h5'
        loaded_model = tf.keras.models.load_model(pth_best_models)
        
        score = loaded_model.evaluate(x_test, y_test, verbose=0)
        los = score[0]
        acc = score[1]
            
        if show:
            print(dir_name[-18:])
            print('\tTest loss      : {:.2%}'.format(score[0]))
            print('\tTest accuracy  : {:.2%}'.format(score[1]))

        if acc>acc_best:
            acc_best = acc
            obj = {
                'model' : loaded_model,
                'dir'   : pth_best_models,
                'score' : score
            }
            best = obj
    print('\nBest model loaded')
    
    return best

def print_best_model(best):
    print('#---- Best Model ----#')
    print('\t Dir       :',best['dir'])
    print('\t Loss      : {:.2%}'.format(best['score'][0]))
    print('\t Accuracy  : {:.2%}'.format(best['score'][1]))
    print('\n')
    best['model'].summary()

def tag_now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

def mkdir(path):
    os.makedirs(path, mode=0o750, exist_ok=True)

def plot_some_values(x, y, n):
    
    plt.figure(figsize=(15,6))
    for i in range(n):
        #Subplot
        plt.subplot(1, 10, i+1)
        
        #plot
        idx = random.randint(0, n)
        img = x[idx]
        lbl = y[idx]
        plt.imshow(img, cmap='binary')
        plt.title(lbl)
        plt.axis('off')

def plot_history(history):
    history_df = pd.DataFrame(history.history)
    
    history_df.loc[:, ['loss', 'val_loss']].plot()
    print("Last Validation Loss: {:0.4f}".format(history_df['val_loss'].iloc[-1]))
    
    history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
    print("Last Validation Accuracy: {:.2%}".format(history_df['val_accuracy'].iloc[-1]))

def get_wrong_predictions(x_test, y_test, y_pred):
    #return a list of indices of all wrong predictions (y_test vs y_pred)
    n = len(y_pred)
    y_wrong = []

    for i in range(n):
        if y_pred[i] != y_test[i]:
            y_wrong.append(i)

    return y_wrong

def plot_wrong_pred(x_test, y_test, y_pred):
    n = len(y_pred)
    y_wrong = []

    for i in range(n):
        if y_pred[i] != y_test[i]:
            y_wrong.append(i)

    acc = 1 - len(y_wrong)/len(y_test)
    print('\naccuracy = {:.2%}  || len(test) = {}  || len(wrong) = {}'.format(acc, len(y_test), len(y_wrong)))


    plt.figure(figsize=(15,6))

    for i in range(10):
        #get idx
        rand_int = random.randint(0, len(y_wrong))
        idx = y_wrong[rand_int]

        #plot
        plt.subplot(1, 10, i+1)
        plt.imshow(x_test[idx], cmap='binary')
        plt.axis('off')
        ttl = str(y_pred[idx]) + ' (' + str(y_test[idx]) + ')'
        plt.title(ttl)

def save_h5_dataset(x_train, y_train, x_test, y_test, filename):
        
    # ---- Create h5 file
    with h5py.File(filename, "w") as f:
        f.create_dataset("x_train", data=x_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("x_test",  data=x_test)
        f.create_dataset("y_test",  data=y_test)
        
    # ---- done
    size=os.path.getsize(filename)/(1024*1024)
    print('Dataset : {:24s} size : {:6.1f} Mo   (saved)'.format(filename, size))

def check_h5_dataset(filename):
    
    with  h5py.File(filename,'r') as f:
        x_tmp_train = f['x_train'][:]
        y_tmp_train = f['y_train'][:]
        x_tmp_test  = f['x_test'][:]
        y_tmp_test  = f['y_test'][:]

        print("\ndataset loaded from h5 file.")
        print(f'x_train : {x_tmp_train.shape}')
        print(f'y_train : {y_tmp_train.shape}')
        print(f'x_test  : {x_tmp_test.shape}')
        print(f'y_test  : {y_tmp_test.shape}')

def load_h5_dataset(filename):
    
    with  h5py.File(filename,'r') as f:
        x_tmp_train = f['x_train'][:]
        y_tmp_train = f['y_train'][:]
        x_tmp_test  = f['x_test'][:]
        y_tmp_test  = f['y_test'][:]

        print("\ndataset loaded from h5 file.")
        print(f'x_train : {x_tmp_train.shape}')
        print(f'y_train : {y_tmp_train.shape}')
        print(f'x_test  : {x_tmp_test.shape}')
        print(f'y_test  : {y_tmp_test.shape}')
    
    return (x_tmp_train, y_tmp_train), (x_tmp_test, y_tmp_test)

def load_h5_dataset_train_test_eval(filename):
    
    with  h5py.File(filename,'r') as f:
        x_tmp_train = f['x_train'][:]
        y_tmp_train = f['y_train'][:]
        x_tmp_test  = f['x_test'][:]
        y_tmp_test  = f['y_test'][:]
        x_tmp_eval  = f['x_eval'][:]
        y_tmp_eval  = f['y_eval'][:]

        print("\ndataset loaded from h5 file.")
        print(f'x_train : {x_tmp_train.shape}')
        print(f'y_train : {y_tmp_train.shape}')
        print(f'x_test  : {x_tmp_test.shape}')
        print(f'y_test  : {y_tmp_test.shape}')
        print(f'x_test  : {x_tmp_eval.shape}')
        print(f'y_test  : {y_tmp_eval.shape}')
    
    return (x_tmp_train, y_tmp_train), (x_tmp_test, y_tmp_test), (x_tmp_eval, y_tmp_eval)