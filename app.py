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
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

ADMIN_CREDENTIALS = {"admin": "1234"}  # Update for security!

# ------------------- EMAIL ALERTS -------------------
EMAIL_SENDER = "youremail@gmail.com"
EMAIL_PASSWORD = "yourapppassword"

def send_email_alert(subject, content, receiver="receiver@gmail.com"):
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
        st.warning(f"Email failed: {e}")

# ------------------- FACE CAPTURE -------------------
def save_face(name):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(f'{DATA_DIR}/haarcascade_frontalface_default.xml')
    faces_data = []
    i = 0

    st.info("Capturing... press 'Q' to finish early.")

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) <= 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        cv2.imshow("Adding Face", frame)
        if cv2.waitKey(1) == ord('q') or len(faces_data) == 100:
            break

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data).reshape(100, -1)

    names_path = os.path.join(DATA_DIR, 'names.pkl')
    faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')

    names = [name] * 100
    if os.path.exists(names_path):
        with open(names_path, 'rb') as f:
            existing_names = pickle.load(f)
        names = existing_names + names

    with open(names_path, 'wb') as f:
        pickle.dump(names, f)

    if os.path.exists(faces_path):
        with open(faces_path, 'rb') as f:
            existing_faces = pickle.load(f)
        faces_data = np.append(existing_faces, faces_data, axis=0)

    with open(faces_path, 'wb') as f:
        pickle.dump(faces_data, f)

    st.success(f"Face added for {name}")
    send_email_alert("New Registration", f"New face added for {name}")

# ------------------- ATTENDANCE -------------------
def mark_attendance():
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(f'{DATA_DIR}/haarcascade_frontalface_default.xml')

    with open(f'{DATA_DIR}/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open(f'{DATA_DIR}/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)
    existing_names = []

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)[0]

            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
            file_path = os.path.join(ATTENDANCE_DIR, f'Attendance_{date}.csv')
            exist = os.path.isfile(file_path)

            if output not in existing_names:
                if not exist:
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["NAME", "TIME"])
                with open(file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([output, timestamp])
                existing_names.append(output)
                st.success(f"Attendance marked for {output}")
                send_email_alert("Attendance Marked", f"{output} at {timestamp}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, output, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# ------------------- VIEW ATTENDANCE -------------------
def view_attendance():
    date = st.date_input("📅 Select Date")
    file_path = os.path.join(ATTENDANCE_DIR, f"Attendance_{date.strftime('%d-%m-%Y')}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.dataframe(df)

        if st.button("📥 Export as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Attendance - {date}", ln=True, align='C')
            for i in range(len(df)):
                pdf.cell(200, 10, txt=f"{df.iloc[i, 0]} - {df.iloc[i, 1]}", ln=True)
            pdf.output("Attendance_Report.pdf")
            st.success("PDF downloaded as Attendance_Report.pdf")
    else:
        st.warning("No data for selected date.")

# ------------------- DASHBOARD -------------------
def show_dashboard():
    all_records = []
    for file in os.listdir(ATTENDANCE_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(ATTENDANCE_DIR, file))
            if "NAME" in df.columns:
                all_records.extend(df["NAME"].tolist())
    if all_records:
        df = pd.DataFrame(all_records, columns=["Name"])
        count = df["Name"].value_counts().reset_index()
        count.columns = ["Name", "Count"]
        st.subheader("📊 Attendance Analytics")
        st.bar_chart(data=count.set_index("Name"))
    else:
        st.info("No attendance records to display.")

# ------------------- STREAMLIT APP -------------------
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
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")
else:
    menu = st.sidebar.radio("📂 Menu", ["Add Face", "Mark Attendance", "View Attendance", "Dashboard", "Logout"])

    if menu == "Add Face":
        name = st.text_input("Enter Name")
        if st.button("Capture Face"):
            if name.strip():
                save_face(name.strip())
            else:
                st.warning("Please enter a valid name.")

    elif menu == "Mark Attendance":
        st.info("Start camera and press 'Q' to stop.")
        if st.button("Start Attendance"):
            mark_attendance()

    elif menu == "View Attendance":
        view_attendance()

    elif menu == "Dashboard":
        show_dashboard()

    elif menu == "Logout":
        st.session_state["admin_logged_in"] = False
        st.success("Logged out")
        st.rerun()
