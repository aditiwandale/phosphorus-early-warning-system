import tensorflow as tf

# Load old model (your local environment)
model = tf.keras.models.load_model("lstm_risk_model.h5", compile=False)

# Save in new Keras format (recommended)
model.save("lstm_risk_model.keras")
