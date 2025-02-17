import numpy as np
import matplotlib.pyplot as plt
import streamlit as st # для создания веб-приложения
from PIL import Image # для работы с разными расширениями картинок
import tensorflow as tf
from tensorflow.keras.datasets import cifar10 # набор данных (60К данных 10 классов)
from tensorflow.keras.models import Sequential # класс для создания слоев нейросети
from tensorflow.keras.layers import Flatten, Dense # класс для создания полносвязных слоев нейросети
from tensorflow.keras.utils import to_categorical # преобразует вектор в матрицу двоичных классов
import ssl

# Отключаем проверку SSL
ssl._create_default_https_context = ssl._create_unverified_context

def create_model():
    # распаковка на тренировочную-валидационную-тестовую выборки
    (x_train, y_train), (x_val, y_val) = cifar10.load_data()
    # подготовка изображений чтобы все пиксели имели вес от 0-1
    x_train = x_train / 255
    x_val = x_val / 255

    # мерки классов преобразовываем в категориальные переменные
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)

    # нейросеть
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)), # принимает на вход изображения 32*32 на 3 канала: R,G,B
        Dense(1000, activation='relu'), # полносвязный слой с функцией активации релу
        Dense(10, activation='softmax') # 10 - нейронов, потому что 10 классов, каждый из 10 нейронов будет активирован от 0-100: вероятность принадлежности к классу
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # лос-вычисляет насколько наше распределение вероятностей далеко от правды
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_val, y_val)) # тренируем массив, batch_size - пачка картинок для тренировки, валидационные данные для проверки
    model.save('veights_neural.h5')


# Обучаем модель, только если запускается файл
create_model()