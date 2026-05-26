import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Select features
X = df[['Pclass', 'Age', 'Fare']]
y = df['Survived']

# Handle missing values
X['Age'] = X['Age'].fillna(X['Age'].mean())

# Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),

    tf.keras.layers.Dense(
        8,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        4,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        1,
        activation='sigmoid'
    )
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=50
)

# Save model
model.save("titanic_ann_model.keras")

print("Model Saved Successfully")