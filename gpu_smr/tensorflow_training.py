import tensorflow as tf

def main():
    """
    Builds, trains, and evaluates a simple neural network on the MNIST dataset.
    """
    print(f"Using TensorFlow version: {tf.__version__}")

    # Step 1: Verify GPU Availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to True to prevent TensorFlow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Successfully configured {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found. The model will train on the CPU.")

    # Step 2: Load and Prepare the MNIST Dataset
    # MNIST is a dataset of 60,000 28x28 grayscale images of handwritten digits (0-9).
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the pixel values from the 0-255 range to the 0-1 range.
    # This helps the network train more effectively.
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # Step 3: Build the Neural Network Model
    # We will create a simple sequential model.
    model = tf.keras.models.Sequential([
        # Flattens the 28x28 image into a 1D array of 784 pixels.
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # A dense layer with 128 neurons and a ReLU activation function.
        tf.keras.layers.Dense(128, activation='relu'),
        # A dropout layer to prevent overfitting. It randomly sets 20% of neuron outputs to zero.
        tf.keras.layers.Dropout(0.2),
        # The final output layer with 10 neurons (one for each digit, 0-9).
        # Softmax converts the outputs to a probability distribution.
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Step 4: Compile the Model
    # This configures the model for training.
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    # Step 5: Train the Model
    print("\n--- Starting Model Training ---")
    # The fit method trains the model for a fixed number of epochs (iterations over the dataset).
    # If a GPU is present, this is the step where it will be heavily used.
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    print("--- Model Training Finished ---\n")

    # Step 6: Evaluate the Model's Performance
    print("--- Evaluating Model Performance ---")
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    main()

