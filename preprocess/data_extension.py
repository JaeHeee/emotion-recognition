import numpy as np
np.random.seed(5)
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# 데이터셋 불러오기 약간씩 변형을 준다.
data_aug_gen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,
                                  zoom_range=0.1,
                                  horizontal_flip = True,
                                  fill_mode='nearest')

data_aug_gen2 = ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,
                                  zoom_range=0.1,
                                  horizontal_flip = True,
                                  fill_mode='nearest')

data_aug_gen3 = ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,
                                  zoom_range=0.1,
                                  horizontal_flip = True,
                                  fill_mode='nearest')

data_aug_gen4 = ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,
                                  zoom_range=0.1,
                                  horizontal_flip = True,
                                  fill_mode='nearest')

data_aug_gen5 = ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,
                                  zoom_range=0.1,
                                  horizontal_flip = True,
                                  fill_mode='nearest')

#data 폴더에 미리 happ,sad,surp,ang,neut 만들어 놓는다. 이곳으로 생성된다.
#/data/FER2013Valid의 각 폴더(Happy,Sad,Surprise,Angry,Neutral)안에 폴더 하나를 만들어서 그곳에 사진들을 모두 넣는다.

valid_generator = data_aug_gen.flow_from_directory('./data/FER2013Valid/Happy',save_to_dir='data/happ',
        target_size=(128, 128),save_prefix='fer', save_format='png',
        color_mode="grayscale",
        batch_size=5,
        class_mode='categorical')

valid_generator2 = data_aug_gen2.flow_from_directory('./data/FER2013Valid/Sad',save_to_dir='data/sad',
        target_size=(128, 128),save_prefix='fer', save_format='png',
        color_mode="grayscale",
        batch_size=6,
        class_mode='categorical')
valid_generator3 = data_aug_gen3.flow_from_directory('./data/FER2013Valid/Surprise',save_to_dir='data/surp',
        target_size=(128, 128),save_prefix='fer', save_format='png',
        color_mode="grayscale",
        batch_size=4,
        class_mode='categorical')
valid_generator4 = data_aug_gen4.flow_from_directory('./data/FER2013Valid/Angry',save_to_dir='data/ang',
        target_size=(128, 128),save_prefix='fer', save_format='png',
        color_mode="grayscale",
        batch_size=8,
        class_mode='categorical')

valid_generator5 = data_aug_gen5.flow_from_directory('./data/FER2013Valid/Neutral',save_to_dir='data/neut',
        target_size=(128, 128),save_prefix='fer', save_format='png',
        color_mode="grayscale",
        batch_size=8,
        class_mode='categorical')

i = 0
for batch in valid_generator:
    i += 1
    if i > 129:
        break

j = 0
for batch in valid_generator2:
    j += 1
    if j > 856:
        break

k = 0
for batch in valid_generator3:
    k += 1
    if k > 1067:
        break

n = 0
for batch in valid_generator4:
    n += 1
    if n > 752:
        break

m = 0
for batch in valid_generator5:
    n += 1
    if n > 752:
        break
