import cv2
import os
from flask import Flask, request, render_template
from datetime import datetime,date
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

nimgs = 10

imgBackground=cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    # Define the time ranges
    time_ranges = [
    (datetime.strptime("12:00:00", "%H:%M:%S").time(), datetime.strptime("14:59:00", "%H:%M:%S").time()),
    (datetime.strptime("18:31:00", "%H:%M:%S").time(), datetime.strptime("19:53:00", "%H:%M:%S").time()),
    (datetime.strptime("14:00:00", "%H:%M:%S").time(), datetime.strptime("15:00:00", "%H:%M:%S").time())
    ]   


    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    current_time_str = datetime.strptime(current_time, "%H:%M:%S").time()
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    attendance_marking = False
    #find the corresponding row of the person
    person_row = df.loc[df['Name']==username]
    time = ""

     # Check if the current time is within any of the specified time ranges
    for start_time, end_time in time_ranges:
         if start_time <= current_time_str <= end_time:
             attendance_marking = True
             period_start_time = start_time

    #check if any rows were found for the person
    print(start_time)
    if not person_row.empty and attendance_marking:
        for index, row in person_row.iterrows():
            username = row['Name']
            roll = row['Roll']
            presence_time = row['Time']
            if not presence_time.startswith(str(period_start_time)[:3]) and len(person_row)<2:
                print (len(person_row))
                with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                    f.write(f'\n{username},{userid},{current_time}')
            elif index == 1 and not presence_time.startswith(str(period_start_time)[:3]):
                with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                    f.write(f'\n{username},{userid},{current_time}')
            elif index == 2 and not presence_time.startswith(str(period_start_time)[:3]):
                with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                    f.write(f'\n{username},{userid},{current_time}')
            else:
                names, rolls, times, l = extract_attendance()
                return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, error="USER ALREADY MARK PRESENT FOR THIS PERIOD")
    elif person_row.empty and attendance_marking: 
        print("no")
        if int(userid) not in df['Roll'].values:
            with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')
    else:
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, error="IT'S NOT TIME TO TAKE ATTENDANCE")
            

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.') 
    ret = True
    cap = cv2.VideoCapture(0)
    #address = "http://192.168.96.174:8080/video"
    #cap.open(address)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, f'{identified_person}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)



@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    count = 0
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    #address = "http://192.168.96.174:8080/video"
    #cap.open(address)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                for imgname in os.listdir(f'static/faces/{newusername}_{newuserid}'):
                    count +=1
                name = newusername+'_'+str(count)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

if __name__ == '__main__':
    app.run(debug=True)
