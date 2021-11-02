import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

data = np.load('02vector.npz')
features = data['features']
target = data['target']
cnn_model = tf.keras.models.load_model('cnn_model.h5', compile=False)
lstm_model = tf.keras.models.load_model('lstm_model.h5', compile=False)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_pred = cnn_model.predict(X_train[:, :, np.newaxis])
train_pred = np.argmax(train_pred, -1)
print("===================CNN Train metrics===================")
print(classification_report(y_train, train_pred))

test_pred = cnn_model.predict(X_test[:, :, np.newaxis])
test_pred = np.argmax(test_pred, -1)
print("===================CNN Test metrics===================")
print(classification_report(y_test, test_pred))

train_pred = lstm_model.predict(X_train[:, np.newaxis, :])
train_pred = np.argmax(train_pred, -1)
print("==================LSTM Train metrics===================")
print(classification_report(y_train, train_pred))

test_pred = lstm_model.predict(X_test[:, np.newaxis, :])
test_pred = np.argmax(test_pred, -1)
print("==================LSTM Test metrics===================")
print(classification_report(y_test, test_pred))
