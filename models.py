'''
Keras models architectures for smile and open mouth recognition.
'''

# Keras / Tensorflow
import tensorflow as tf
from tensorflow import keras
# from keras_preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import Sequential, Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv2D, Flatten, MaxPool2D, Dense, Dropout
from tensorflow.keras.layers import Activation, BatchNormalization, ZeroPadding2D, Lambda, Concatenate
keras.backend.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from ipywidgets import IntProgress



class TrainProgressBar(keras.callbacks.Callback):
    '''
    Display training progress by epochs.
    '''
    def __init__(self, max_value, min_value=1):
        # Initialize progress bar
        self.bar = IntProgress(
            value=min_value,
            min=min_value,
            max=max_value,
            step=1,
            description='Initializing...'
        )
        display(self.bar)
    def on_epoch_end(self, epoch, logs):
        # Update values
        self.bar.value = epoch+1
        self.bar.description = '[{:>{tab}} / {}]'.format(epoch+1, self.bar.max, tab=len(str(self.bar.max)))
    def on_train_end(self, logs=None):
        # Change bar color after training
        self.bar.bar_style = 'success'
    

def FeedForward(input_shape, output_shape, layers_dims, dropout_rate=0.5):
    '''
    Returns Feed-Forward neural network Keras Model().
    Number of hidden layers and neurons are specified in layer_dims.
    
    Parameters
    ----------
    input_shape : int
        Number of input features.
    output_shape : int
        Number of ouput features.
    layer_dims : list of ints
        Hidden layer sizes.
    dropout_rate : float in [0, 1] or None
        Rate for dropout layers. Skip Dropout if None.
    
    Returns
    -------
    model : Keras Model() object
    '''
    
    # Input layer
    X_input = Input(shape=input_shape, name='input')
    
    # Add hidden layers
    for i in range(len(layers_dims)):
        layer = Dense(layers_dims[i], activation='relu', name='fc_' + str(i+1))
        if i == 0:
            X = layer(X_input)
        else:
            X = layer(X)
        if dropout_rate is not None:
            X = Dropout(dropout_rate, name='drop_' + str(i+1))(X) 
    
    # Output layer
    output_layer = Dense(output_shape, activation='softmax', name='output')
    if len(layers_dims) == 0:
        output = output_layer(X_input)
    else:
        output = output_layer(X)
    
    model =  Model(inputs=X_input, outputs=output, name='FeedForward_{}_layers'.format(len(layers_dims)))
    return model


def FeedForward_VGG16_combined(input_shapes, output_shape, layers_dims, dropout_rates):
    '''
    Returns Feed-Forward neural network Keras Model().
    Networks takes as inputs VGG16 FC1 features and pairwise landmarks distances.
    Number of hidden layers and neurons are specified in layer_dims.
    layer_dims should be a dictionary with:
        'vgg' : list of layers dims for VGG path.
        'landmarks' : list of layers dims for landmarks path.
        'combined' : list of layers dims for combined path.
    
    Parameters
    ----------
    input_shape : int
        Number of input features shapes.
    output_shape : int
        Number of ouput features.
    layer_dims : dict of layers_dims
        Hidden layer sizes.
    dropout_rate : float in [0, 1] or None
        Rate for dropout layers. Skip Dropout if None.
    
    Returns
    -------
    model : Keras Model() object
    '''
    
    def get_layers(X_input, layers_dims, name_prefix, dropout_rate):
        X = X_input
        for i in range(len(layers_dims)):
            layer = Dense(layers_dims[i], activation='relu', name='fc_' + name_prefix + '_' + str(i+1))
            X = layer(X)
            if dropout_rate is not None:
                X = Dropout(dropout_rate, name='drop_' + name_prefix + '_' + str(i+1))(X)
        return X
    
    # Get layers dims
    vgg_layers_dims = layers_dims['vgg']
    landmarks_layers_dims = layers_dims['landmarks']
    combined_layers_dims = layers_dims['combined']
    
    # Get dropouts
    vgg_drop_rate = dropout_rates['vgg']
    landmarks_drop_rate = dropout_rates['landmarks']
    combined_drop_rate = dropout_rates['combined']
    
    # Inputs layer
    vgg_input = Input(shape=input_shapes['vgg'], name='vgg_input')
    landmarks_input = Input(shape=input_shapes['landmarks'], name='landmarks_input')
    
    # Add VGG path hidden layers
    X_vgg = get_layers(vgg_input, vgg_layers_dims, 'vgg', vgg_drop_rate)
    # Add landmarks path hidden layers
    X_landmarks = get_layers(landmarks_input, landmarks_layers_dims, 'landmarks', landmarks_drop_rate)
    # Concatenate paths
    X = Concatenate()([X_vgg, X_landmarks])
    # Add combined path hidden layers
    X = get_layers(X, combined_layers_dims, 'combined', combined_drop_rate)
    
    output = Dense(output_shape, activation='softmax', name='output')(X)
    
    model =  Model(inputs=[vgg_input, landmarks_input], outputs=output, name='FeedForward_Combined')
    return model


def extract_features(model, filenames, path_to_images, display_bar=True):
    '''
    Extracts output from model on list of filenames.
    
    Parameters
    ----------
    model : Keras model.
    filenames : list
        List of filenames to process.
    path_to_images : path to folder with images
    
    Returns
    -------
    features : ndarray
        Extracted features.
    '''
    if display_bar:
        bar = IntProgress(value=1, min=1, max=len(filenames), step=1, description='Initializing...')
        display(bar)

    features = np.zeros((len(filenames), 4096), dtype=np.float)
    for i in range(len(filenames)):
        img_path = os.path.join(path_to_images, filenames[i])
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features[i] = feature_extractor.predict(x).squeeze()
        if display_bar:
            bar.value = i+1
            bar.description = '[{:>{tab}} / {}]'.format(i+1, bar.max, tab=len(str(bar.max)))
    if display_bar:
        bar.bar_style = 'success'
    return features


def main():
    pass


if __name__ == '__main__':
    main()