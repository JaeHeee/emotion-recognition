emotion-recognition(remind)
===========================
카메라를 이용하여, 얼굴을 검출하고, 감정인식을 진행합니다.


Installation
------------
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

Dataset
-----------
fer2013 - https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data 
<br>
fer2013 plus - https://github.com/Microsoft/FERPlus

```
python generate_data.py -d <basefolder> -fer <fer2013.csv path> -ferplus <fer2013new.csv path>
```

Deep Learnig Model
------------------
https://github.com/gitshanks/fer2013 이곳의 모델을 사용하였습니다.
