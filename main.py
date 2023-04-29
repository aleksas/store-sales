from data import X_train, y_train, X_val, y_val, scaled_test,y_test

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, Flatten

def transformer_encoder(inputs, num_heads, dff, rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = Dense(dff, activation='relu')(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return out2

# Parameters
num_heads = 8
dff = 128
dropout_rate = 0.1

# Define model architecture
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = transformer_encoder(inputs, num_heads=num_heads, dff=dff, rate=dropout_rate)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(dropout_rate)(x)
outputs = Dense(1, activation='linear')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), shuffle=False)

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequence = data[i:i + window_size]
        sequences.append(sequence)
    return np.array(sequences)

# Parameters
window_size = 15

# Create sequences for test data
X_test_sequences = create_sequences(scaled_test, window_size)

# Make predictions
y_pred = model.predict(X_test_sequences)

# Compute evaluation metric (e.g., mean absolute error)
from sklearn.metrics import mean_absolute_error

# You need to provide the true test labels (y_test) to evaluate the model
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Absolute Error on Test Data: {mae}')