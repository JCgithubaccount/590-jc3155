import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, layers, regularizers
from tensorflow.keras.callbacks import CSVLogger

epochs = 20

data = np.load('clean01_processed.npz')
features = data['features']
target = data['target']
model = SentenceTransformer('all-mpnet-base-v2')
features = np.apply_along_axis(model.encode, 0, features)
np.savez('02vector.npz', features=features, target=target)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

cnn_logger = CSVLogger('cnn.log', append=True, separator=';')
cnn_model = Sequential()
cnn_model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l=0.05)))
cnn_model.add(layers.Dense(3, activation='softmax'))
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
history1 = cnn_model.fit(X_train[:, :, np.newaxis], y_train,
                         epochs=epochs,
                         batch_size=32,
                         validation_data=(X_test[:, :, np.newaxis], y_test), callbacks=[cnn_logger])
cnn_model.save('cnn_model.h5')

train_acc = history1.history['acc']
val_acc = history1.history['val_acc']
train_loss = history1.history['loss']
val_loss = history1.history['val_loss']

plt.plot(range(epochs), train_acc, label='train acc')
plt.plot(range(epochs), val_acc, label='val acc')
plt.title('train acc vs val acc')
plt.legend()
plt.savefig('cnn_accuracy.png')

plt.plot(range(epochs), train_loss, label='train loss')
plt.plot(range(epochs), val_loss, label='val loss')
plt.title('train loss vs val loss')
plt.legend()
plt.savefig('cnn_loss.png')

lstm_logger = CSVLogger('lstm.log', append=True, separator=';')
lstm_model = Sequential()
lstm_model.add(layers.LSTM(32, activation='tanh', recurrent_activation='sigmoid'))
lstm_model.add(layers.Flatten())
lstm_model.add(layers.Dense(32, activation='tanh', kernel_regularizer=regularizers.l1_l2(0.02, 0.05)))
lstm_model.add(layers.Dropout(0.8))
lstm_model.add(layers.Dense(3, activation='softmax'))
lstm_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['acc'])

history2 = lstm_model.fit(X_train[:, np.newaxis, :], y_train,
                          epochs=epochs,
                          batch_size=32,
                          validation_data=(X_test[:, np.newaxis, :], y_test), callbacks=[lstm_logger])
lstm_model.save('lstm_model.h5')

train_acc = history2.history['acc']
val_acc = history2.history['val_acc']
train_loss = history2.history['loss']
val_loss = history2.history['val_loss']

plt.plot(range(epochs), train_acc, label='train acc')
plt.plot(range(epochs), val_acc, label='val acc')
plt.title('train acc vs val acc')
plt.legend()
plt.savefig('lstm_accuracy.png')

plt.plot(range(epochs), train_loss, label='train loss')
plt.plot(range(epochs), val_loss, label='val loss')
plt.title('train loss vs val loss')
plt.legend()
plt.savefig('lstm_loss.png')
