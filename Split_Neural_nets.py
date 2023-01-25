import tensorflow as tf
import wandb
from tensorflow import keras
from tensorflow.keras import layers 
import numpy as np
import time

wandb.init(project="test-project", entity="team-research")


num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

#TODO: 
# split datasets into num of clients (separate the categories for each client)
# try to have different targets for each, some overlap is fine 
 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

config = {
              "learning_rate": 0.001,
              "epochs": 10,
              "batch_size": 32,
              "log_step": 200,
              "val_log_step": 50,
              "architecture": "CNN",
              "dataset": "MNIST"
           }
config = wandb.config

client_model = keras.Sequential(
    [
        #1st layer 
        keras.Input(shape=input_shape),
        #2nd
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        #3rd
        layers.MaxPooling2D(pool_size=(2, 2)),
    ]
)

server_model = keras.Sequential(
    [
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

loss_fn = tf.keras.losses.CategoricalCrossentropy()
client_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
server_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

n_epochs = 5
batch_size = 32
n_steps = len(x_train) // batch_size
n_steps_test = len(x_test) // batch_size
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_accuracy = tf.keras.metrics.CategoricalAccuracy()

@tf.function
def test_step(x, y):
    val_logits = server_model(client_model(x), training = False)
    val_loss = loss_fn(y, val_logits)
    val_accuracy.update_state(y, val_logits)
    return val_loss


@tf.function
def train_step(x, y):
    #per epoch 
    client_output = client_model(x)
    server_input = tf.identity(client_output)
    server_output = server_model(server_input)
    loss = loss_fn(server_output, y)
    server_gradient = tf.gradients(loss, server_model.trainable_variables)
    server_optimizer.apply_gradients(zip(server_gradient, server_model.trainable_variables))
    server_input_grad = tf.gradients(loss, server_input)
    # grad of loss wrt to clinet paramenters
    # loss wrt to clinet output(servber_i8nput) 
    client_gradient = tf.gradients(client_output, client_model.trainable_variables, grad_ys=server_input_grad)
    client_optimizer.apply_gradients(zip(client_gradient, client_model.trainable_variables))

    #loss calculations
    train_acc_metric.update_state(y, server_output)
    return loss



# for epoch in range(n_epochs):
#     print("\nStart of epoch %d" % (epoch,))
#     start_time = time.time()

#     for step in range(1, n_steps + 1):
#         X_batch, y_batch = random_batch(x_train, y_train)
#         loss_value = train_step(X_batch, y_batch)

#         if step % 200 == 0:
#             print(
#                 "Training loss (for one batch) at step %d: %.4f"
#                 % (step, float(loss_value))
#             )
#             print("Seen so far: %d samples" % ((step + 1) * batch_size))


#     train_acc = train_acc_metric.result()
#     print("Training acc over epoch: %.4f" % (float(train_acc),))
#     train_acc_metric.reset_states()


#     for step in range(1, n_steps + 1):
#         x_batch_test, y_batch_test = random_batch(x_test, y_test)
#         test_step(x_batch_test, y_batch_test)

#     val_acc = val_accuracy.result()
#     val_accuracy.reset_states()
#     print("Validation acc: %.4f" % (float(val_acc),))
#     print("Time taken: %.2fs" % (time.time() - start_time))


    wandb.log({'epochs': epoch,
                'loss': np.mean(loss_value),
                'train_acc': float(train_acc), 
                'val_loss': np.mean(test_step(x_batch_test, y_batch_test)),
                'val_acc':float(val_acc)})


# for multiple clients 

# aggregate the activations func output from client 
# wait for the better clinet accurtacy by comapring the loss function of the clients 
# server decides which clients will be processed accordingly 
# agg vs sequential 
client_num = 5 
client_model_list = [client_model for clients in range(client_num)]
client_opt_list = [client_optimizer for clients in range(client_num)]

total_epochs = 2

for epoch in range(total_epochs):
    for client_n in range(client_num):
        print('Current client {}'.format(client_n))
        client = client_model_list[client_n]
        client_opt = client_opt_list[client_n]


        if client_n == 0:
            if epoch != 0:
                prev_client = client_num - 1 
                prev_client_weights = client_model_list[prev_client].get_weights()
                client.set_weights(prev_client_weights)
                print('Loaded!!')
        else:
            prev_client = client_n-1
            prev_client_weights = client_model_list[prev_client].get_weights()
            client.set_weights(prev_client_weights)
            print('Loaded!!')



        for step in range(1, n_steps + 1):
            X_batch, y_batch = random_batch(x_train, y_train)
            loss_value = train_step(X_batch, y_batch)

            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        train_acc_metric.reset_states()

        for step in range(1, n_steps + 1):
            x_batch_test, y_batch_test = random_batch(x_test, y_test)
            test_step(x_batch_test, y_batch_test)

        val_acc = val_accuracy.result()
        val_accuracy.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))

        wandb.log({'epochs': epoch,
                'loss': np.mean(loss_value),
                'train_acc': float(train_acc), 
                'val_loss': np.mean(test_step(x_batch_test, y_batch_test)),
                'val_acc':float(val_acc)})
