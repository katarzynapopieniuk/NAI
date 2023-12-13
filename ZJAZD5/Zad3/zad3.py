# TYTUŁ: NEURAL NETWORKS FOR CLASSIFICATION
#
# AUTORZY: Katarzyna Popieniuk s22048 i Jakub Styn s22449
#
# OPIS PROBLEMU:
# 1. Naucz sieć rozpoznawać ubrania. np. GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database.
#
# INSTRUKCJA PRZYGOTOWANIA ŚRODOWISKA
# 1. Zainstalować interpreter python w wersji 3+ oraz narzędzie pip
# 2. Pobrać projekt
# 3. Uruchomić wybraną konsolę/terminal
# 4. Zainstalować wymagane biblioteki za pomocą komend:
# pip install tensorflow
# pip install silence_tensorflow
# 5. Przejść do ścieżki z projektem (w systemie linux komenda cd)
# 6. Uruchomić projekt przy pomocy polecenia:
# python .\zad3.py


from silence_tensorflow import silence_tensorflow

import tensorflow as tf

silence_tensorflow()

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

def recognize_clothes(model):
    """
        Train and evaluate a neural network model to recognize clothes.

        This function compiles, trains, and evaluates a neural network model to recognize clothing items using the Fashion MNIST dataset.

        Parameters:
        model (tf.keras.Sequential): Neural network model architecture.

        Returns:
        test_acc (float): Accuracy of the trained model on the test dataset.
        """

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    return test_acc

if __name__ == '__main__':
    """
        Perform clothing recognition using neural networks.

        This script loads the Fashion MNIST dataset, creates and evaluates two neural network models to recognize clothing items, and displays their test accuracies.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    test_acc1 = recognize_clothes(model)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    test_acc2 = recognize_clothes(model)
    print('\nTest accuracy for small neuron network:', test_acc1)
    print('\nTest accuracy for bigger neuron network:', test_acc2)
