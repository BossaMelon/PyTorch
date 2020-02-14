from tensorflow.python.keras import Input, callbacks
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

def create_conv_ad_model(input_shape):
    
    ### a few convolutions followed by a dense layer to predict a flat sensor output
    
    width, height = input_shape[:2]
    m_input = Input(shape=(input_shape))
    l = m_input
    conv_layers = [8, 16, 32]
    for i in conv_layers:
        l = Conv2D(i, 3, padding='same', activation='relu', name='conv_{}'.format(i))(l)
        l = MaxPooling2D(pool_size=2, name='pool_{}'.format(i))(l)
        l = Dropout(0.1)(l, training=False)
    latent_space = Flatten(name='flatten')(l)
    m_output = Dense(width * height, activation='sigmoid', name='finaldense')(latent_space)

    model = Model(m_input, m_output)
    encoder = Model(m_input, latent_space)
    

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-2), metrics=['mse'])
    model.summary()
    return model, encoder

def plot_keras_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training loss', 'validation loss'], loc='upper right')
    plt.show()

