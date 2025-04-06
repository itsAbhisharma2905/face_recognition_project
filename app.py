# app.py
import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import cv2
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import time
import csv
from fpdf import FPDF
import smtplib
from email.message import EmailMessage

# ------------------- CONFIG -------------------
DATA_DIR = 'data'
ATTENDANCE_DIR = 'Attendance'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

ADMIN_CREDENTIALS = {"admin": "1234"}  # 🔐 Change this!
EMAIL_SENDER = "youremail@gmail.com"   # 🔐 Replace
EMAIL_PASSWORD = "your-app-password"   # 🔐 Replace with app password

# ------------------- EMAIL -------------------
def send_email_alert(subject, content, receiver="receiveremail@gmail.com"):
    try:
        msg = EmailMessage()
        msg.set_content(content)
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = receiver

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
    except Exception as e:
        st.warning(f"Failed to send email: {e}")

# ------------------- FACE SAVING -------------------
def save_face(name):
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        st.error("Webcam not accessible. Please check your camera.")
        return

    facedetect = cv2.CascadeClassifier(CASCADE_PATH)
    faces_data = []
    i = 0

    st.info("Capturing face. Press 'Q' in the webcam window to stop.")

    while True:
        ret, frame = video.read()
        if not ret or frame is None:
            st.error("Failed to capture video frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) < 100 and i % 5 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(faces_data)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capturing Face", frame)
        if cv2.waitKey(1) == ord('q') or len(faces_data) == 100:
            break

    video.release()
    cv2.destroyAllWindows()

    if not faces_data:
        st.warning("No faces were captured.")
        return

    faces_data = np.asarray(faces_data).reshape(len(faces_data), -1)

    # Save name and face data
    names_path = os.path.join(DATA_DIR, 'names.pkl')
    faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')

    if os.path.exists(names_path):
        with open(names_path, 'rb') as f:
            names = pickle.load(f)
    else:
        names = []

    names.extend([name] * len(faces_data))
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)

    if os.path.exists(faces_path):
        with open(faces_path, 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
    else:
        faces = faces_data

    with open(faces_path, 'wb') as f:
        pickle.dump(faces, f)

    st.success(f"{name}'s face saved successfully.")
    send_email_alert("New Face Registered", f"New face added for {name}.")

# ------------------- ATTENDANCE -------------------
def mark_attendance():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        st.error("Webcam not accessible.")
        return

    facedetect = cv2.CascadeClassifier(CASCADE_PATH)

    with open(f'{DATA_DIR}/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open(f'{DATA_DIR}/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    marked = []

    st.info("Press 'Q' in webcam window to stop recognition.")

    while True:
        ret, frame = video.read()
        if not ret or frame is None:
            st.warning("Frame not captured.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)[0]

            date_str = datetime.now().strftime("%d-%m-%Y")
            time_str = datetime.now().strftime("%H:%M:%S")
            file_path = os.path.join(ATTENDANCE_DIR, f'Attendance_{date_str}.csv')
            if output not in marked:
                if not os.path.exists(file_path):
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["NAME", "TIME"])
                with open(file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([output, time_str])
                marked.append(output)
                st.success(f"Marked {output} at {time_str}")
                send_email_alert("Attendance Marked", f"{output} marked present at {time_str}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(frame, output, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# ------------------- VIEW & PDF -------------------
def view_attendance():
    date = st.date_input("Select Date")
    file_path = os.path.join(ATTENDANCE_DIR, f"Attendance_{date.strftime('%d-%m-%Y')}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.dataframe(df)
        if st.button("Export to PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for i in range(len(df)):
                pdf.cell(200, 10, txt=f"{df.iloc[i, 0]} - {df.iloc[i, 1]}", ln=True)
            pdf.output("Attendance_Report.pdf")
            st.success("Exported to Attendance_Report.pdf")
    else:
        st.warning("No attendance record for the selected date.")

# ------------------- DASHBOARD -------------------
def show_dashboard():
    all_names = []
    for file in os.listdir(ATTENDANCE_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(ATTENDANCE_DIR, file))
            all_names.extend(df["NAME"].tolist())
    if all_names:
        count = pd.Series(all_names).value_counts().reset_index()
        count.columns = ["Name", "Count"]
        st.subheader("📊 Attendance Summary")
        st.bar_chart(data=count.set_index("Name"))
    else:
        st.info("No attendance data found.")

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="📸 Smart Attendance", layout="centered")
st.title("📸 Smart Face Recognition Attendance System")

if "admin_logged_in" not in st.session_state:
    st.session_state["admin_logged_in"] = False

if not st.session_state["admin_logged_in"]:
    st.subheader("🔐 Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
            st.session_state["admin_logged_in"] = True
            st.rerun()
        else:
            st.error("Invalid credentials")
else:
    menu = st.sidebar.selectbox("Menu", ["Add Face", "Mark Attendance", "View Attendance", "Dashboard", "Logout"])

    if menu == "Add Face":
        name = st.text_input("Enter Name")
        if st.button("Capture Face"):
            if name.strip():
                save_face(name.strip())
            else:
                st.warning("Enter a valid name.")

    elif menu == "Mark Attendance":
        st.warning("Make sure your webcam is connected. Press 'Q' to stop.")
        if st.button("Start Recognition"):
            mark_attendance()

    elif menu == "View Attendance":
        view_attendance()

    elif menu == "Dashboard":
        show_dashboard()

    elif menu == "Logout":
        st.session_state["admin_logged_in"] = False
        st.rerun()
