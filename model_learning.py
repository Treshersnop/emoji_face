import tensorflow as tf
import keras
from keras import Sequential, layers


batch_size = 256  # размер мини-выборки
image_size = (48, 48)  # размер изображений
dataset_url = 'F:/Programs/RO/train'
dataset_test_url = 'F:/Programs/RO/test'


# сохранение названий классов
def save_classe(classes):
    with open(r'F:/Programs/RO/classes.txt', 'w') as fp:
        for item in classes:
            # write each item on a new line
            fp.write("%s\n" % item)


if __name__ == '__main__':
    print(tf.__version__)
    # данные, на которых нейросеть будет учиться
    train_ds = keras.utils.image_dataset_from_directory(dataset_url,
                                                        subset='training',
                                                        seed=42,
                                                        validation_split=0.1,
                                                        batch_size=batch_size,
                                                        image_size=image_size)
    validation_ds = keras.utils.image_dataset_from_directory(dataset_url,
                                                             subset='validation',
                                                             seed=42,
                                                             validation_split=0.1,
                                                             batch_size=batch_size,
                                                             image_size=image_size)
    class_names = train_ds.class_names
    save_classe(class_names)
    # данные, на которых нейросеть проверит качество
    test_ds = keras.utils.image_dataset_from_directory(dataset_test_url,
                                                       batch_size=batch_size,
                                                       image_size=image_size)
    # для ускорения обучения
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    print('обучение модели')
    model = Sequential()
    model.add(layers.Conv2D(16, (5, 5), padding='same', activation='relu', input_shape=(48, 48, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (5, 5), padding='same', activation='relu'))
    print('связывание слоев')
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    print('выходной слой')
    model.add(layers.Dense(7, activation='softmax'))
    print('компилирование модели')
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print('обучение модели')
    history = model.fit(train_ds, validation_data=validation_ds,
                        epochs=5, verbose=2)
    print('оценить качество обучения')
    scores = model.evaluate(test_ds, verbose=1)
    print('Доля верных ответов', round((scores[1] * 100), 4))
    print('сохранение модели')
    model.save('emotions.h5')
    print('сохранение модели на локальный комп')
    keras.models.load_model('emotions.h5')



