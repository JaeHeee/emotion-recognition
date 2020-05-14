from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 데이터셋 불러오기 약간씩 변형을 준다.
data_aug_gen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,
                                  zoom_range=0.1,
                                  horizontal_flip = True,
                                  fill_mode='nearest')


#data 폴더에 미리 happy,sad,surprise,angry,neutral  이곳으로 생성된다.
#/data/FER2013Valid의 각 폴더(Happy,Sad,Surprise,Angry,Neutral)안에 폴더 하나를 만들어서 그곳에 사진들을 모두 넣는다.

valid_generator_happy = data_aug_gen.flow_from_directory('./data/FER2013Valid/Happy',save_to_dir='data/happy',
        target_size=(128, 128),save_prefix='fer', save_format='png',
        color_mode="grayscale",
        batch_size=5,
        class_mode='categorical')

valid_generator_sad = data_aug_gen.flow_from_directory('./data/FER2013Valid/Sad',save_to_dir='data/sad',
        target_size=(128, 128),save_prefix='fer', save_format='png',
        color_mode="grayscale",
        batch_size=6,
        class_mode='categorical')

valid_generator_surprise = data_aug_gen.flow_from_directory('./data/FER2013Valid/Surprise',save_to_dir='data/surprise',
        target_size=(128, 128),save_prefix='fer', save_format='png',
        color_mode="grayscale",
        batch_size=4,
        class_mode='categorical')

valid_generator_angry = data_aug_gen.flow_from_directory('./data/FER2013Valid/Angry',save_to_dir='data/angry',
        target_size=(128, 128),save_prefix='fer', save_format='png',
        color_mode="grayscale",
        batch_size=8,
        class_mode='categorical')

valid_generator_neutral = data_aug_gen.flow_from_directory('./data/FER2013Valid/Neutral',save_to_dir='data/neutral',
        target_size=(128, 128),save_prefix='fer', save_format='png',
        color_mode="grayscale",
        batch_size=8,
        class_mode='categorical')


i = 0
while i <= 129:
    for batch in valid_generator_happy:
        i += 1
j = 0
while j <= 856:
    for batch in valid_generator_sad:
        j += 1

k = 0
while k <= 1067:
    for batch in valid_generator_surprise:
        k += 1

n = 0
while n <= 752:
    for batch in valid_generator_angry:
        n += 1

m = 0
while m <= 752:
    for batch in valid_generator_neutral:
        m += 1
