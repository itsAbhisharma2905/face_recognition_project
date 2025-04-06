import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import os
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import csv
from sklearn.neighbors import KNeighborsClassifier
from fpdf import FPDF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# === SETTINGS ===
DATA_DIR = "data"
ATTENDANCE_DIR = "Attendance"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# === CONFIG ===
st.set_page_config(page_title="Face Recognition System", layout="centered")

# === AUTH ===
ADMIN_USERNAME = "Abhi"
ADMIN_PASSWORD = "2905"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# === EMAIL FUNCTION ===
def send_email_alert(subject, body):
    sender_email = "your_email@gmail.com"
    receiver_email = "receiver_email@gmail.com"
    password = "your_app_password"  # Use App Password for Gmail

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        server.quit()
    except Exception as e:
        st.warning(f"Failed to send email: {e}")

# === VIDEO PROCESSOR FOR FACE CAPTURE ===
class FaceCapture(VideoProcessorBase):
    def __init__(self):
        self.frames = []
        self.name = None
        self.capture_enabled = False
        self.facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if img is None or img.size == 0:
            return av.VideoFrame.from_ndarray(np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24")

        if self.capture_enabled and self.name:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = cv2.resize(img[y:y + h, x:x + w], (50, 50))
                if len(self.frames) < 100:
                    self.frames.append(face)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# === SAVE FACE TO FILE ===
def save_face_to_data(name, faces_data):
    faces_data = np.asarray(faces_data).reshape(len(faces_data), -1)
    name_path = f"{DATA_DIR}/names.pkl"
    faces_path = f"{DATA_DIR}/faces_data.pkl"

    names = []
    faces = np.empty((0, 7500))

    if os.path.exists(name_path):
        with open(name_path, "rb") as f:
            names = pickle.load(f)

    if os.path.exists(faces_path):
        with open(faces_path, "rb") as f:
            faces = pickle.load(f)

    names += [name] * len(faces_data)
    faces = np.append(faces, faces_data, axis=0)

    with open(name_path, "wb") as f:
        pickle.dump(names, f)
    with open(faces_path, "wb") as f:
        pickle.dump(faces, f)

    send_email_alert("Face Registered", f"{name}'s face has been saved to the system.")

# === ATTENDANCE MARKING ===
def mark_attendance_live():
    with open(f"{DATA_DIR}/names.pkl", "rb") as f:
        labels = pickle.load(f)
    with open(f"{DATA_DIR}/faces_data.pkl", "rb") as f:
        faces_data = pickle.load(f)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces_data, labels)

    class AttendanceProcessor(VideoProcessorBase):
        def __init__(self):
            self.facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.marked = set()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop = img[y:y+h, x:x+w]
                resized = cv2.resize(crop, (50, 50)).flatten().reshape(1, -1)
                pred = knn.predict(resized)[0]
                timestamp = datetime.now().strftime("%H:%M:%S")
                date = datetime.now().strftime("%d-%m-%Y")
                csv_path = f"{ATTENDANCE_DIR}/Attendance_{date}.csv"

                if pred not in self.marked:
                    self.marked.add(pred)
                    if not os.path.exists(csv_path):
                        with open(csv_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["NAME", "TIME"])
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([pred, timestamp])
                    send_email_alert("Attendance Marked", f"{pred} marked present at {timestamp}.")

                cv2.putText(img, pred, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="attendance", video_processor_factory=AttendanceProcessor)

# === ATTENDANCE DASHBOARD ===
def show_dashboard():
    st.header("📊 Attendance Dashboard")

    selected_date = st.date_input("Select Date", datetime.today())
    date_str = selected_date.strftime("%d-%m-%Y")
    file_path = f"{ATTENDANCE_DIR}/Attendance_{date_str}.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.dataframe(df)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📁 Export to Excel"):
                df.to_excel(f"Attendance_{date_str}.xlsx", index=False)
                st.success("Exported to Excel!")

        with col2:
            if st.button("🧾 Export to PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Attendance Report - {date_str}", ln=True, align='C')
                for i in range(len(df)):
                    row = df.iloc[i]
                    pdf.cell(200, 10, txt=f"{row['NAME']} - {row['TIME']}", ln=True)
                pdf.output(f"Attendance_{date_str}.pdf")
                st.success("Exported to PDF!")
    else:
        st.warning("No attendance found for this date.")

# === MAIN UI ===
# ... [rest of your code above remains the same] ...

# === MAIN INTERFACE ===
if not st.session_state.logged_in:
    st.title("🔐 Admin Login")
    uname = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if uname == ADMIN_USERNAME and pwd == ADMIN_PASSWORD:
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid credentials")
else:
    st.sidebar.title("📋 Menu")
    option = st.sidebar.radio("Select Option", ["📤 Upload Face Data", "📝 Mark Attendance", "📊 Dashboard", "🚪 Logout"])

    if option == "📤 Upload Face Data":
        st.title("Upload Face Data (Cloud Friendly)")
        uploaded_faces = st.file_uploader("Upload `faces_data.pkl`", type=["pkl"])
        uploaded_names = st.file_uploader("Upload `names.pkl`", type=["pkl"])

        if uploaded_faces and uploaded_names:
            with open(os.path.join(DATA_DIR, "faces_data.pkl"), "wb") as f:
                f.write(uploaded_faces.getbuffer())
            with open(os.path.join(DATA_DIR, "names.pkl"), "wb") as f:
                f.write(uploaded_names.getbuffer())
            st.success("Face data uploaded successfully!")

    elif option == "📝 Mark Attendance":
        st.title("Real-Time Attendance")
        try:
            mark_attendance_live()
        except Exception as e:
            st.error(f"Webcam not accessible. Error: {e}")
            st.info("Try running locally for real-time webcam access.")

    elif option == "📊 Dashboard":
        show_dashboard()

    elif option == "🚪 Logout":
        st.session_state.logged_in = False
        st.success("Logged out successfully.")
        st.rerun()
