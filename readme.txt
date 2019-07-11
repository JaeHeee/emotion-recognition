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

<generate_data.py>
data 폴더를 우선 생성한다.
data 폴더안에, FER2013Test,FER2013Valid,FER2013Train폴더 만들고, 각 폴더별로 0~9 감정폴더 만들어놓고 파일 실행

python generate_data.py -d ./data -fer ./fer2013.csv -ferplus ./fer2013new.csv

0 - angry 
1 - disgust
2 - fear
3 - Happy
4 - sad
5 - surprise
6 - neutral
7 - contempt
8 - unknown
9 - X

Neutral, Happy, Angry, Sad, Surprise 5가지 감정이용. 
나머지 감정 사진들은 모두 삭제하고 0,3,4,5,6 폴더를 각각 Angry, Happy, Sad, Surprise, Neutral로 이름을 변경해준다.

<data_extension.py>
data 폴더에 미리 happ,sad,surp,ang,neut 만들어 놓는다. 이곳으로 생성된다.
/data/FER2013Valid의 각 폴더(Happy,Sad,Surprise,Angry,Neutral)안에 폴더 하나를 만들어서 그곳에 사진들을 모두 넣는다.
생성된 사진들을  FER2013Train,FER2013Test,FER2013Valid 폴더로 적절히 분류한다.

python data_extension.py

<face_emotion_recognition.py>
python face_emotion_recognition.py

adadelta(lr=0.03, rho=0.95, epsilon=1e-08)
steps_per_epoch=1666, epochs=30

acc = 75.19%