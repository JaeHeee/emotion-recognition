# emotion-recognition(remind)
=============================
webcam을 이용하여, 얼굴을 검출하고, 감정인식을 진행합니다.


### Installation
----------------
- numpy
- matplotlib
- pandas
- pydotplus
- h5py
- scikit-learn
- scipy
- dlib
- imutils
- OpenCV
- tensorflow
- keras

### Dataset
-----------
fer2013 - https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
fer2013 plus - https://github.com/Microsoft/FERPlus

```
python generate_data.py -d <basefolder> -fer <fer2013.csv path> -ferplus <fer2013new.csv path>
```

### Deep Learnig Model
----------------------
