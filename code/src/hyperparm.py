from tqdm import tqdm
import itertools
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

def hyperparameter_tuning(X_train_tfidf, y_train, X_val_tfidf, y_val):
    num_epochs = 100
    num_classes = 6
    vocab_size = X_train_tfidf.shape[1]
    dropout_rates = [0.2, 0.3, 0.4, 0.5]
    activations_list = ['relu', 'leaky_relu', 'elu', 'tanh']
    num_neurons = [32, 64, 128, 256]
    opt_lr = [1e-2, 1e-3, 1e-4]
    L2_lr = [1e-2, 1e-3, 1e-4]
    best_params = None
    best_val_acc = 0

    hyperparam_combi = itertools.product(dropout_rates, num_neurons, activations_list, opt_lr, L2_lr)

    for dr, nn, al, olr, l2lr in tqdm(hyperparam_combi):
        modelTest = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(vocab_size,)),
        tf.keras.layers.Dense(nn*2, activation=al),
        tf.keras.layers.Dropout(dr),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(nn, activation=al),
        tf.keras.layers.Dropout(dr),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(nn//2, activation=al),
        tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-2), activation='softmax')
        ])

        optimizer = tf.keras.optimizers.legacy.Adam(olr)
        modelTest.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=0,
            restore_best_weights=True
        )
        histTest = modelTest.fit(
            X_train_tfidf, y_train,
            batch_size=16,
            epochs=num_epochs,
            validation_data=(X_val_tfidf, y_val),
            verbose=0,
            callbacks=[early_stop,]
        )

        val_acc = max(histTest.history['val_accuracy'])
        # print(f"Dropout: {dr}, Activation: {al}, Hidden Units: {nn}, L2 Reg: {l2lr}, LR: {olr}, Best Val Acc: {val_acc}\n===========================")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (dr, nn, al, olr, l2lr)
    print(f"Final Best Hyperparameters: Dropout: {best_params[0]},\nActivation: {best_params[2]},\nHidden Units: {best_params[1]},\nL2 Reg: {best_params[4]},\nLR: {best_params[3]},\nBest Val Acc: {best_val_acc}")
    return