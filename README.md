🎯 Face Recognition Attendance System with Streamlit
A modern, secure, and real-time face recognition attendance system with an intuitive Streamlit interface. It uses machine learning (KNN) for face recognition and supports webcam integration directly in-browser using Streamlit-WebRTC, making it 100% deployable on Streamlit Cloud. 🌐💻

🚀 Features
✅ Real-time face detection and recognition
✅ Add new faces with names
✅ Admin login system
✅ Attendance marking with timestamp
✅ Prevents duplicate attendance entries
✅ View attendance records with date filters
✅ Analytics dashboard (most frequent attendees, stats)
✅ Export attendance as Excel/PDF
✅ Built-in email alerts for successful attendance
✅ Beautiful responsive UI with dark mode toggle
✅ Streamlit-compatible webcam support (via streamlit-webrtc)

🧠 Tech Stack

Python 🐍

OpenCV (Face Detection & Processing)

scikit-learn (KNN Classifier)

Streamlit (Web UI)

streamlit-webrtc (In-browser webcam)

Pandas (Data Handling)

fpdf (PDF export)

smtplib (Email alerts)

Matplotlib (Charts)

💡 How it Works

Admin Login: Secure access using username & password.

Add Face: Capture your face and store 100 facial samples.

Train KNN Model: Automatically updates when new data is added.

Mark Attendance: Recognizes face and logs attendance with time.

Analytics: View records, filter by date, and export to Excel/PDF.

Email Notification: Sends alerts after attendance is marked.


🔐 Admin Credentials
Username	Password
admin	      1234


📂 Project Structure

📁 Face-Attendance-App/
├── app.py
├── requirements.txt
├── background.png (optional)
├── 📁 data/
│   ├── haarcascade_frontalface_default.xml
│   ├── faces_data.pkl
│   └── names.pkl
├── 📁 Attendance/
│   └── Attendance_<date>.csv


🤝 Contributing
Feel free to fork, enhance, or suggest new features! Pull requests are welcome 🚀

🏆 Author
Abhi-abhi.sharma2905@gmail.com
📧 Feel free to connect for collaboration, project ideas, or contributions!

⭐ If you like it...
Give this repo a ⭐ and share it with your friends!
Let's build cool stuff together! 💡🔥
