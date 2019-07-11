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
data ������ �켱 �����Ѵ�.
data �����ȿ�, FER2013Test,FER2013Valid,FER2013Train���� �����, �� �������� 0~9 �������� �������� ���� ����

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

Neutral, Happy, Angry, Sad, Surprise 5���� �����̿�. 
������ ���� �������� ��� �����ϰ� 0,3,4,5,6 ������ ���� Angry, Happy, Sad, Surprise, Neutral�� �̸��� �������ش�.

<data_extension.py>
data ������ �̸� happ,sad,surp,ang,neut ����� ���´�. �̰����� �����ȴ�.
/data/FER2013Valid�� �� ����(Happy,Sad,Surprise,Angry,Neutral)�ȿ� ���� �ϳ��� ���� �װ��� �������� ��� �ִ´�.
������ ��������  FER2013Train,FER2013Test,FER2013Valid ������ ������ �з��Ѵ�.

python data_extension.py

<face_emotion_recognition.py>
python face_emotion_recognition.py

adadelta(lr=0.03, rho=0.95, epsilon=1e-08)
steps_per_epoch=1666, epochs=30

acc = 75.19%