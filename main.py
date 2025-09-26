# Tensor
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
from IPython import display

# Misc
import numpy as np
import matplotlib.pyplot as plt

# Properties
DATA_PATH = "data/"
visualize_example_waveform = True
visualize_example_spectrogram_and_waveform = True
visualize_example_spectrograms = True
visualize_training = True

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def get_spectrogram(waveform):
    # Convert waveform into spectrogram
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    
    # Get magnitude
    spectrogram = tf.abs(spectrogram)

    # Add a channels dimension
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram    

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram > 2):
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio,label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

if __name__ == "__main__":
    print("Welcome.")

    # Load in data
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=DATA_PATH,
        batch_size=64,
        validation_split=0.2,
        seed=0,
        output_sequence_length=32000,
        subset='both'
    )

    label_names = np.array(train_ds.class_names)


    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    for example_audio, example_labels in train_ds.take(1):
        # Visualize example waveforms
        if visualize_example_waveform:
            plt.figure(figsize=(16, 10))
            rows, cols = 3, 3
            n = rows * cols
            for i in range(n):
                plt.subplot(rows, cols, i+1)
                audio_signal = example_audio[i].numpy()
                plt.plot(audio_signal)
                plt.title(label_names[example_labels[i]])
                plt.yticks(np.arange(-1.2, 1.2, 0.2))
                plt.ylim([-1.1, 1.1])
            plt.show()

        # Print shape of example spectrograms
        for i in range(3):
            label = label_names[example_labels[i]]
            waveform = example_audio[i]
            spectrogram = get_spectrogram(waveform)

            print("Label: ", label)
            print("Waveform shape:", waveform.shape)
            print("spectrogram shape: ", spectrogram.shape)
            print("Audio playback")
            
            # Not really useful for terminal
            display.display(display.Audio(waveform, rate=32000))

            # Vizual example of spectrogram
            if visualize_example_spectrogram_and_waveform:
                fig, axes = plt.subplots(2, figsize=(12, 8))
                timescale = np.arange(waveform.shape[0])
                axes[0].plot(timescale, waveform.numpy())
                axes[0].set_title('Waveform')
                axes[0].set_xlim([0, 32000])

                plot_spectrogram(spectrogram.numpy(), axes[1])
                axes[1].set_title('spectrogram')
                plt.suptitle(label.title())
                plt.show()


    train_spectrogram_ds = make_spec_ds(train_ds)
    val_spectrogram_ds = make_spec_ds(val_ds)
    test_spectrogram_ds = make_spec_ds(test_ds)

    # Visualize example spectrograms
    for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
        break
    
    if visualize_example_spectrograms:
        rows = 3
        cols = 3
        n = rows*cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

        for i in range(n):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            plot_spectrogram(example_spectrograms[i].numpy(), ax)
            ax.set_title(label_names[example_spect_labels[i].numpy()])

            plt.show()

    train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

    input_shape = example_spectrograms.shape[1:]
    print('Input shape:', input_shape)
    num_labels = len(label_names)

    # Instantiate the 'tf.keras.layers.Normalization' layer
    norm_layer = layers.Normalization()
    norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input
        layers.Resizing(64,64),
        
        # Normalize
        norm_layer,
        
        # Conv layers
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),

        # Regularization
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = 10
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2, restore_best_weights=True)
    )

    if visualize_training:
        metrics = history.history
        plt.figure(figsize=(16,6))
        plt.subplot(1,2,1)
        plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.ylim([0, max(plt.ylim())])
        plt.xlabel('Epoch')
        plt.ylabel('Loss [CrossEntropy]')

        plt.subplot(1,2,2)
        plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
        plt.legend(['accuracy', 'val_accuracy'])
        plt.ylim([0, 100])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy [%]')
        plt.show()

    # Evalutate the model
    model.evaluate(test_spectrogram_ds, return_dict=True)

    plot_model(model, show_shapes=True, show_layer_names=True, to_file='images/model.png')
    