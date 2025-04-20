import tkinter as tk
from tkinter import *
from tkinter import simpledialog
from tkinter import filedialog
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint 
import pickle
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import requests
from PIL import Image, ImageTk
import io
import tkinter.font as tkfont
import pygame

class Preloader:
    def __init__(self, gif_url, main):
        self.main = main
        self.preloader = None
        self.gif_url = gif_url
        self.frames = []
        self.current_frame = 0

    def show(self):
        self.preloader = Toplevel(self.main)
        self.preloader.title("Loading...")
        self.preloader.attributes('-fullscreen', True)
        self.preloader.configure(bg='#2C3E50')

        response = requests.get(self.gif_url)
        img_data = io.BytesIO(response.content)
        gif = Image.open(img_data)

        for frame in range(gif.n_frames):
            gif.seek(frame)
            self.frames.append(ImageTk.PhotoImage(gif.resize((self.preloader.winfo_screenwidth(), 
                                                            self.preloader.winfo_screenheight()), Image.LANCZOS)))

        self.label = tk.Label(self.preloader, image=self.frames[self.current_frame], bg='#2C3E50')
        self.label.pack(expand=True)

        self.update_gif()
        self.preloader.after(20000, self.close)
        self.preloader.after(5000, self.play_audio)
          # Schedule audio to play 5 seconds after preloader starts
        #self.preloader.after(5000, self.play_audio)


    def update_gif(self):
        self.current_frame += 1
        if self.current_frame >= len(self.frames):
            self.current_frame = 0
        self.label.config(image=self.frames[self.current_frame])
        self.preloader.after(100, self.update_gif)

    def close(self):
        self.preloader.destroy()
        self.main.deiconify()
    def play_audio(self):
         # Play the MP3 audio after preloader closes
        pygame.mixer.init()
        pygame.mixer.music.load("loading_audio.mp3")
        pygame.mixer.music.play()

# Define all main application functions here (omitted for brevity, include loadData, datasetProcessing, etc., as in original)
def loadData():
    global dataset, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', tk.END)
    text.insert(tk.END, filename + " dataset loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(tk.END, str(dataset.head()))
    labels = np.unique(dataset['classification'])

    label = dataset.groupby('classification').size()
    label.plot(kind="bar")
    plt.xlabel("Chronic Kidney Disease Type")
    plt.ylabel("Count")
    plt.title("Chronic Kidney Disease Graph")
    plt.show()


def datasetProcessing():
    text.delete('1.0', tk.END)
    global dataset, label_encoder, scaler, X, Y, X_train, y_train, X_test, y_test
    dataset.fillna(0, inplace = True)

    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric 
            label_encoder.append(le)
    text.insert(tk.END,"Dataset after preprocessing\n\n")
    text.insert(tk.END,str(dataset)+"\n\n")
    dataset = dataset.values
    X = dataset[:,1:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    Y = Y.astype(int)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(tk.END, "Dataset Train & Test Splits\n")
    text.insert(tk.END, "Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(tk.END, "80% dataset used for training  : "+str(X_train.shape[0])+"\n")
    text.insert(tk.END, "20% dataset user for testing   : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(tk.END, algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(tk.END, algorithm+" Precision : "+str(p)+"\n")
    text.insert(tk.END, algorithm+" Recall    : "+str(r)+"\n")
    text.insert(tk.END, algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()        

def runCNN():
    text.delete('1.0', tk. END)
    global accuracy, precision, recall, fscore, cnn_model
    global X_train, y_train, X_test, y_test
    accuracy = []
    precision = []
    recall = [] 
    fscore = []

    cnn_model = Sequential()
    #adding CNN layer wit 32 filters to optimized dataset features using 32 neurons
    cnn_model.add(Convolution2D(32, (1, 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    #adding maxpooling layer to collect filtered relevant features from previous CNN layer
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #adding another CNN layer to further filtered features
    cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #collect relevant filtered features
    cnn_model.add(Flatten())
    #defining output layers
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    #defining prediction layer with Y target data
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile the CNN with LSTM model
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #train and load the model
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size = 8, epochs = 1, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("CNN", testY, predict)
    
def runLSTM():
    global X_train, y_train, X_test, y_test
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2] * X_train.shape[3]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2] * X_test.shape[3]))
    lstm_model = Sequential()#defining deep learning sequential object
    #adding LSTM layer with 100 filters to filter given input X train data to select relevant features
    lstm_model.add(LSTM(100,input_shape=(X_train.shape[1], X_train.shape[2])))
    #adding dropout layer to remove irrelevant features
    lstm_model.add(Dropout(0.5))
    #adding another layer
    lstm_model.add(Dense(100, activation='relu'))
    #defining output layer for prediction
    lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
    #compile LSTM model
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #start training model on train data and perform validation on test data
    #train and load the model
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm_model.fit(X_train, y_train, batch_size = 8, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lstm_model.load_weights("model/lstm_weights.hdf5")
    predict = lstm_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("LSTM", testY, predict)

def runEnsemble():
    global X_train, y_train, X_test, y_test, cnn_model, Y
    ensemble_model = Model(cnn_model.inputs, cnn_model.layers[-2].output)#creating cnn model
    cnn_features = ensemble_model.predict(X)  #extracting cnn features from test data
    Y = np.argmax(Y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(cnn_features, Y, test_size=0.2)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    calculateMetrics("Ensemble CNN with Random Forest", y_test, predict)
    

def graph():
    df = pd.DataFrame([['CNN','Accuracy',accuracy[0]],['CNN','Precision',precision[0]],['CNN','Recall',recall[0]],['CNN','FSCORE',fscore[0]],
                       ['LSTM','Accuracy',accuracy[1]],['LSTM','Precision',precision[1]],['LSTM','Recall',recall[1]],['LSTM','FSCORE',fscore[1]],
                       ['Ensemble CNN with Random Forest','Accuracy',accuracy[2]],['Ensemble CNN with Random Forest','Precision',precision[2]],['Ensemble CNN with Random Forest','Recall',recall[2]],['Ensemble CNN with Random Forest','FSCORE',fscore[2]],
                      ],columns=['Algorithms','Accuracy','Value'])
    df.pivot("Algorithms", "Accuracy", "Value").plot(kind='bar')
    plt.title("All Algorithm Comparison Graph")
    plt.show()

def predictDisease():
    text.delete('1.0', tk.END)
    global label_encoder, scaler, cnn_model, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    temp = dataset.values
    index = 0
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            dataset[columns[i]] = pd.Series(label_encoder[index].transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric 
            index = index + 1
    dataset = dataset.values
    dataset = dataset[:,1:dataset.shape[1]]
    X = scaler.transform(dataset)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    predict = cnn_model.predict(X)
    for i in range(len(predict)):
        pred = np.argmax(predict[i])
        text.insert(END,"Test Data = "+str(temp[i])+" =====> Predicted As "+str(labels[pred])+"\n\n")


# Similarly define other functions (datasetProcessing, calculateMetrics, runCNN, etc.)

def create_main_application():
    global main, text, pathlabel
    main = tk.Tk()
    main.title("Chronic Kidney Disease Prediction")
    main.geometry("1200x700")
    main.configure(bg='#2C3E50')
    main.withdraw()

    title_frame = Frame(main, bg='#2C3E50', height=70, width=1600)
    title_frame.pack_propagate(False)
    title_frame.pack()
    title_frame.place(x=0, y=10)

    canvas = Canvas(title_frame, bg='#2C3E50', highlightthickness=0, width=1600, height=70)
    canvas.pack()

    font = ('times', 20, 'bold')
    text_str = "CHRONIC KIDNEY DISEASE PREDICTION USING CNN, LSTM & ENSEMBLE MODEL"
    text_width = tkfont.Font(family='times', size=20, weight='bold').measure(text_str)

    title_x = [1600]
    title_text = canvas.create_text(title_x[0], 35, text=text_str, fill='white', font=font, anchor='w')

    def animate():
        title_x[0] -= 2
        canvas.coords(title_text, title_x[0], 35)
        if title_x[0] < -text_width:
            title_x[0] = 1600
        main.after(20, animate)
    animate()

    bg_image = Image.open("ChatGPT Image.png")
    screen_width = main.winfo_screenwidth()
    screen_height = main.winfo_screenheight()
    bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    background_label = Label(main, image=bg_photo)
    background_label.image = bg_photo
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    font1 = ('times', 13, 'bold')
    uploadButton = tk.Button(main, text="Upload Chronic Kidney Dataset", command=loadData)
    uploadButton.place(x=50, y=100)
    uploadButton.config(font=font1, fg='white', bg='#2ECC71')

    pathlabel = tk.Label(main)
    pathlabel.config(bg='#1ABC9C', fg='#FFFFFF')
    pathlabel.config(font=font1)
    pathlabel.place(x=460, y=100)

    preprocessButton = tk.Button(main, text="Preprocess Dataset", command=datasetProcessing)
    preprocessButton.place(x=50,y=150)
    preprocessButton.config(font=font1, fg='white', bg='#2ECC71') 

    cnnButton =tk. Button(main, text="Run CNN Algorithm", command=runCNN)
    cnnButton.place(x=330,y=150)
    cnnButton.config(font=font1, fg='white', bg='#2ECC71') 

    lstmButton = tk.Button(main, text="Run LSTM Algorithm", command=runLSTM)
    lstmButton.place(x=630,y=150)
    lstmButton.config(font=font1, fg='white', bg='#2ECC71')

    ensembleButton = tk.Button(main, text="Run Ensemble Random Forest", command=runEnsemble)
    ensembleButton.place(x=50,y=200)
    ensembleButton.config(font=font1, fg='white', bg='#2ECC71') 

    graphButton = tk.Button(main, text="Comparison Graph", command=graph)
    graphButton.place(x=330,y=200)
    graphButton.config(font=font1, fg='white', bg='#2ECC71')

    predictButton = tk.Button(main, text="Predict Disease from Test Data", command=predictDisease)
    predictButton.place(x=630,y=200)
    predictButton.config(font=font1, fg='white', bg='#2ECC71') 


    font1 = ('times', 12, 'bold')


    # Add other buttons and widgets similarly (preprocessButton, cnnButton, etc.)

    text = tk.Text(main, height=25, width=80, bg='#FFFFFF', fg='black', bd=0, highlightthickness=0, font=('times', 12, 'bold'))
    text.place(x=880, y=265)

    #scroll = Scrollbar(main, command=text.yview)
    #text.configure(yscrollcommand=scroll.set)
    #scroll.place(x=1170, y=250, height=355)

    title_frame.lift()
    return main

if __name__ == "__main__":
    gif_url = "https://i.pinimg.com/originals/00/54/5c/00545cb7179c504433d4c8f5e845f286.gif"
    main = create_main_application()
    preloader = Preloader(gif_url, main)
    preloader.show()
    main.mainloop()