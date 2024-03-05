import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Permute, Reshape, Multiply, Lambda, Softmax, Dot, Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Bidirectional, LSTM, Dense, Activation
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer

class SelfAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, x):
        # eij = tanh(Wx + b)
        eij = K.tanh(K.dot(x, self.W) + self.b)
        
        # ai = exp(eij) / sum_j(exp(eij))
        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=2, keepdims=True)  # sum over the features dimension
        
        # weighted input
        weighted_input = x * weights
        return weighted_input

    def compute_output_shape(self, input_shape):
        return input_shape

def vectorize(df_train, df_test):
    df_train['NameDesc'] = df_train['NameDesc'].apply(lambda x: ' '.join(x))
    df_test['NameDesc'] = df_test['NameDesc'].apply(lambda x: ' '.join(x))

    X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['NameDesc']).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(df_test['NameDesc']).toarray()

    y_train = df_train['STRIDE'].values
    y_test = df_test['STRIDE'].values

    return X_train_tfidf, X_test_tfidf, y_train, y_test

# model
def initialise_model(num_classes, vocab_size, lr):
    OPTIMIZER = tf.keras.optimizers.legacy.Adam(lr)

    embedding_dim = 128
    num_filters = 256
    kernel_size = 3
    pool_size = 2

    # Input Layer
    inputs = Input(shape=(vocab_size,))

    # Word Embedding Layer
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = Activation('tanh')(x)

    # Self-attention layer
    attention_output = SelfAttentionLayer()(x)
    x = Concatenate()([attention_output])

    # Convolutional Layers with different kernel sizes
    conv_blocks = []
    for i in range(3):
        conv = Conv1D(filters=num_filters, kernel_size=kernel_size, padding="valid", activation="relu", strides=1)(x)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)

    x = Concatenate()(conv_blocks)

    # BLSTM Layer
    blstm = Bidirectional(LSTM(128, return_sequences=True))(attention_output)
    x = Concatenate()([attention_output, blstm])
    blstm = GlobalMaxPooling1D()(x)

    # concatenate convolutional features and BLSTM features
    x = Concatenate()([x for x in conv_blocks])
    x = Concatenate()([x, blstm])

    # Fully Connected Layer
    x = Dense(128)(x)
    x = Activation('tanh')(x)

    # Output Layer
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # plot_model(model, show_shapes=False, show_layer_names=True)
    return model

tfidf_vectorizer = TfidfVectorizer()
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
    restore_best_weights=True
)

def train_loop(model, X_train_tfidf, y_train, NUM_EPOCHS, BATCH_SIZE):
    hist = model.fit(
        X_train_tfidf, y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        # validation_data=(X_val_tfidf, y_val),
        verbose=1,
        callbacks=[early_stop,]
    )
    return hist, model