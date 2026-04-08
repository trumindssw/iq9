import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

X = np.load("X.npy")
y = np.load("y.npy")

print("Dataset:", X.shape)

# reshape for Conv1D
X = X.reshape(-1,1500,1)

# before reshaping: X shape: (33826, 1500), y shape: (33826,)
# after reshaping: (33826, 1500, 1) this is required for Conv1D.


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def create_model():

    inputs = layers.Input(shape=(1500,1))

    x = layers.Conv1D(64,5,activation="relu",padding="same")(inputs)
    x = layers.Conv1D(128,5,activation="relu",padding="same")(x)
    x = layers.Conv1D(256,3,activation="relu",padding="same")(x)

    x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(128,activation="relu")(x)

    outputs = layers.Dense(9,activation="softmax")(x)

    model = models.Model(inputs,outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


model = create_model()

model.summary()

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=256,
    validation_data=(X_test,y_test)
)

loss,acc = model.evaluate(X_test,y_test)

print("Test accuracy:",acc)

model.export("cnn_dpi_model")

print("Model exported successfully")
