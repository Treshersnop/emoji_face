from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing import image

image_path = 'F:/Programs/RO/demo/demo6.jpg'
img_size = (48, 48)


def show_image(img_path, phrase='Изображение'):
    # загрузка изображения
    img = cv2.imread(img_path)
    # вывод его на экран
    cv2.imshow(phrase, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# загрузка названий классов
def load_classes():
    name_classes = []
    with open(r'F:/Programs/RO/classes.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            name_classes.append(x)
    return name_classes


# определяет какой смайл
def which_emoji(value1, value2, num1, num2):
    emoji_path = 'F:/Programs/RO/emoji/'
    if value1 - value2 >= 0.1:
        return emoji_path+str(num1) + '.jpg'
    elif num1 > num2:
        return emoji_path+str(num1*10 + num2) + '.jpg'
    else:
        emoji_path + str(num2*10 + num1) + '.jpg'


if __name__ == '__main__':
    show_image(image_path)
    # загрузка модели нейросети
    model = load_model('emotions.h5')
    # форматирование загрузка изображения в нейросеть
    img1 = image.load_img(image_path, target_size=img_size)
    x = image.img_to_array(img1)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images)
    # вывод насколько нейросеть распознала эмоции
    print(classes)
    # самая высокая оценка распознавания
    max_class = (max(classes[0]), np.argmax(classes))
    # вторая по величине оценка распознавания
    max_2 = 0
    max_num_2 = 0
    for i in range(len(classes[0])):
        num = classes[0][i]
        if (max_2 <= num) and (max_class[0] != num):
            max_2 = num
            max_num_2 = i
    max_2_class = (max_2, max_num_2)
    # загрузка названий классов
    name_class = load_classes()
    # путь к какому смайлу определила нейросеть
    emoji_path = which_emoji(max_class[0], max_2_class[0], max_class[1], max_2_class[1])
    print(emoji_path)
    show_image(emoji_path, 'Какой вы смайл')
