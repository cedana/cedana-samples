import tensorflow as tf
import os
import time

strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
# Calculate the global batch size. The dataset will yield batches of this size.
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Create a dummy dataset
def create_dataset(num_samples=100000):
    # Create random features and labels
    X = tf.random.normal(shape=(num_samples, 10))
    y = tf.random.uniform(shape=(num_samples,), minval=0, maxval=2, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    # Shuffle and batch the dataset
    return dataset.shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)

train_dataset = create_dataset()


with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

print("Model built and compiled successfully within the MirroredStrategy scope.")
model.summary()


print("\nStarting training...")
start_time = time.time()

model.fit(train_dataset, epochs=15000)

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")
