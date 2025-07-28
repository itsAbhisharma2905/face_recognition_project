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
from io import BytesIO
from PIL import Image

# ------------------- CONFIG -------------------
# Create directories if they don't exist
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

# Download haarcascade file if not exists
HAARCASCADE_PATH = 'data/haarcascade_frontalface_default.xml'
if not os.path.exists(HAARCASCADE_PATH):
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        HAARCASCADE_PATH
    )

# Constants
DATA_DIR = 'data'
ATTENDANCE_DIR = 'Attendance'
ADMIN_CREDENTIALS = {"admin": "1234"}  # Change this in production!

# ------------------- EMAIL -------------------
def send_email_alert(subject, content, receiver="receiveremail@gmail.com"):
    email_sender = st.session_state.get('email_sender', "youremail@gmail.com")
    email_password = st.session_state.get('email_password', "your-app-password") 
    
    # Only send email if credentials are properly configured
    if email_sender == "youremail@gmail.com" or email_password == "your-app-password":
        st.warning("📧 Email not configured. Configure in Settings.")
        return
        
    try:
        msg = EmailMessage()
        msg.set_content(content)
        msg["Subject"] = subject
        msg["From"] = email_sender
        msg["To"] = receiver

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(email_sender, email_password)
            smtp.send_message(msg)
        st.success("📧 Email alert sent successfully!")
    except Exception as e:
        st.error(f"📧 Email not sent: {e}")

# ------------------- FACE CAPTURE -------------------
def save_face(name):
    # Initialize variables
    faces_data = []
    i = 0
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    status_text = st.empty()
    
    # Get webcam
    try:
        # For deployment, use streamlit-webrtc for browser webcam access
        status_text.info("📸 Starting webcam... If prompted, please allow camera access.")
        
        # Simulate face capturing process without actual webcam
        # This is where you'd integrate with streamlit-webrtc in a live deployment
        st.warning("⚠️ In live deployment, this would use your webcam through browser.")
        st.info("ℹ️ For demo purposes, simulating face capture...")
        
        # Create sample face data using random values
        # In a real app, this would be actual face captures
        for _ in range(10):  # Simulate 10 captures
            progress = min(1.0, (i+1) / 10)
            progress_bar.progress(progress)
            status_text.write(f"Capturing: {i+1}/10 images")
            time.sleep(0.5)  # Simulate processing time
            i += 1
        
        # Create sample face data for demonstration
        faces_data = np.random.rand(10, 50*50*3).astype(np.uint8)
        
        # Process and save face data
        names_path = os.path.join(DATA_DIR, 'names.pkl')
        faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')

        if not os.path.exists(names_path):
            names = [name] * len(faces_data)
        else:
            with open(names_path, 'rb') as f:
                names = pickle.load(f)
            names.extend([name] * len(faces_data))

        with open(names_path, 'wb') as f:
            pickle.dump(names, f)

        if not os.path.exists(faces_path):
            faces = faces_data
        else:
            with open(faces_path, 'rb') as f:
                faces = pickle.load(f)
            faces = np.append(faces, faces_data, axis=0)

        with open(faces_path, 'wb') as f:
            pickle.dump(faces, f)

        status_text.empty()
        st.success(f"✅ {name}'s face added with {len(faces_data)} samples.")
        send_email_alert("New Face Added", f"New face data added for {name}.")
        return True
        
    except Exception as e:
        st.error(f"❌ Error capturing face: {e}")
        return False

# ------------------- ATTENDANCE -------------------
def mark_attendance():
    try:
        # Check if face data exists
        if not os.path.exists(f'{DATA_DIR}/names.pkl') or not os.path.exists(f'{DATA_DIR}/faces_data.pkl'):
            st.error("❌ No face data available. Please add faces first.")
            return
            
        with open(f'{DATA_DIR}/names.pkl', 'rb') as f:
            LABELS = pickle.load(f)
        with open(f'{DATA_DIR}/faces_data.pkl', 'rb') as f:
            FACES = pickle.load(f)
            
        # Train the KNN model
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(FACES, LABELS)
        
        # Get current date
        current_date = datetime.now().strftime("%d-%m-%Y")
        file_path = os.path.join(ATTENDANCE_DIR, f'Attendance_{current_date}.csv')
        
        # Load existing attendance to avoid duplicates
        existing_names = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row:
                        existing_names.append(row[0])
        
        # For deployment - this would use streamlit-webrtc for browser webcam
        st.warning("⚠️ In live deployment, this would use your webcam through browser.")
        st.info("ℹ️ For demo purposes, simulating attendance...")
        
        # Create status area
        status_text = st.empty()
        recognition_result = st.empty()
        
        # Simulate detection
        status_text.info("🔍 Scanning for faces...")
        time.sleep(1)
        
        # Get unique names from our database
        unique_names = list(set(LABELS))
        
        # Simulate detecting a random person from our database
        if unique_names:
            # Pick a random name from the database
            import random
            detected_name = random.choice(unique_names)
            
            # Mark attendance
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if detected_name not in existing_names:
                # Create file if doesn't exist
                if not os.path.exists(file_path):
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["NAME", "TIME"])
                        
                # Append attendance
                with open(file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([detected_name, timestamp])
                    
                existing_names.append(detected_name)
                recognition_result.success(f"✅ Attendance marked for {detected_name} at {timestamp}")
                send_email_alert("Attendance Marked", f"{detected_name} marked present at {timestamp}")
            else:
                recognition_result.info(f"ℹ️ {detected_name} already marked present")
        else:
            st.error("❌ No registered faces in the database.")
            
    except Exception as e:
        st.error(f"❌ Error in face recognition: {e}")

# ------------------- VIEW ATTENDANCE -------------------
def view_attendance():
    st.subheader("📅 Select Date to View Attendance")
    
    # Date selection
    date = st.date_input("Select Date", datetime.now())
    formatted_date = date.strftime("%d-%m-%Y")
    file_path = os.path.join(ATTENDANCE_DIR, f"Attendance_{formatted_date}.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # Show data table
        st.subheader(f"📋 Attendance for {formatted_date}")
        st.dataframe(df, use_container_width=True)
        
        # Statistics
        st.subheader("📊 Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Present", len(df))
        with col2:
            st.metric("Unique Attendees", df['NAME'].nunique())
        
        # Export options
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            if st.button("📄 Export to PDF"):
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    
                    # Add title
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(190, 10, f"Attendance Report - {formatted_date}", 0, 1, 'C')
                    pdf.ln(10)
                    
                    # Add table headers
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(95, 10, "Name", 1, 0, 'C')
                    pdf.cell(95, 10, "Time", 1, 1, 'C')
                    
                    # Add table data
                    pdf.set_font("Arial", '', 12)
                    for i in range(len(df)):
                        pdf.cell(95, 10, str(df.iloc[i, 0]), 1, 0, 'L')
                        pdf.cell(95, 10, str(df.iloc[i, 1]), 1, 1, 'L')
                    
                    # Save PDF
                    pdf_path = os.path.join(ATTENDANCE_DIR, f"Attendance_Report_{formatted_date}.pdf")
                    pdf.output(pdf_path)
                    
                    # Create download button
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"Attendance_Report_{formatted_date}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {e}")
        
        with export_col2:
            if st.button("📊 Export to CSV"):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"Attendance_Report_{formatted_date}.csv",
                    mime="text/csv"
                )
    else:
        st.warning(f"⚠️ No attendance data found for {formatted_date}")

# ------------------- DASHBOARD -------------------
def show_dashboard():
    st.subheader("📊 Attendance Analytics")
    
    # Check for attendance files
    attendance_files = [f for f in os.listdir(ATTENDANCE_DIR) if f.startswith("Attendance_") and f.endswith(".csv")]
    
    if not attendance_files:
        st.info("📭 No attendance data available yet.")
        return
        
    # Gather all attendance data
    all_data = []
    dates = []
    
    for file in attendance_files:
        try:
            date = file.replace("Attendance_", "").replace(".csv", "")
            dates.append(date)
            df = pd.read_csv(os.path.join(ATTENDANCE_DIR, file))
            df['DATE'] = date
            all_data.append(df)
        except Exception as e:
            st.warning(f"⚠️ Error reading file {file}: {e}")
    
    if not all_data:
        st.warning("⚠️ No valid attendance data found.")
        return
        
    combined_df = pd.concat(all_data)
    
    # Dashboard tabs
    tab1, tab2, tab3 = st.tabs(["Daily Stats", "Individual Stats", "Overall Stats"])
    
    # Tab 1: Daily Stats
    with tab1:
        st.subheader("📆 Daily Attendance")
        
        # Date selector
        selected_date = st.selectbox("Select Date", sorted(dates, reverse=True))
        
        date_df = combined_df[combined_df['DATE'] == selected_date]
        
        if not date_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Attendance", len(date_df))
            with col2:
                st.metric("Unique Attendees", date_df['NAME'].nunique())
                
            # Time distribution
            try:
                # Convert time to hours for histogram
                date_df['HOUR'] = pd.to_datetime(date_df['TIME'], format="%H:%M:%S").dt.hour
                hour_count = date_df['HOUR'].value_counts().sort_index()
                
                st.subheader("⏰ Attendance Time Distribution")
                st.bar_chart(hour_count)
            except Exception as e:
                st.error(f"❌ Error processing time data: {e}")
        else:
            st.warning(f"⚠️ No data available for {selected_date}")
    
    # Tab 2: Individual Stats
    with tab2:
        st.subheader("👤 Individual Attendance")
        
        # Get unique names
        all_names = sorted(combined_df['NAME'].unique())
        
        if all_names:
            selected_name = st.selectbox("Select Person", all_names)
            
            person_df = combined_df[combined_df['NAME'] == selected_name]
            
            # Count attendance by date
            attendance_count = person_df.groupby('DATE').size().reset_index(name='COUNT')
            
            st.metric("Total Days Present", len(attendance_count))
            
            # Attendance history
            st.subheader("📅 Attendance History")
            st.dataframe(person_df[['DATE', 'TIME']], use_container_width=True)
        else:
            st.warning("⚠️ No attendance data available")
    
    # Tab 3: Overall Stats
    with tab3:
        st.subheader("🔍 Overall Statistics")
        
        # Attendance frequency by person
        name_counts = combined_df['NAME'].value_counts()
        
        st.subheader("👥 Attendance Frequency")
        st.bar_chart(name_counts)
        
        # Most regular attendees
        st.subheader("🏆 Most Regular Attendees")
        st.dataframe(
            name_counts.reset_index().rename(
                columns={'index': 'Name', 'NAME': 'Days Present'}
            ).head(5),
            use_container_width=True
        )

# ------------------- USER MANAGEMENT -------------------
def user_management():
    st.subheader("👥 User Management")
    
    # Check if face data exists
    names_path = os.path.join(DATA_DIR, 'names.pkl')
    faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')
    
    if not os.path.exists(names_path) or not os.path.exists(faces_path):
        st.warning("⚠️ No user data available.")
        return
    
    try:
        # Load user data
        with open(names_path, 'rb') as f:
            names = pickle.load(f)
        
        # Get unique names and count
        unique_names = {}
        for name in names:
            if name in unique_names:
                unique_names[name] += 1
            else:
                unique_names[name] = 1
        
        # Display users
        st.subheader(f"🧑‍💼 Registered Users: {len(unique_names)}")
        
        user_df = pd.DataFrame({
            'Name': list(unique_names.keys()),
            'Samples': list(unique_names.values())
        })
        
        st.dataframe(user_df, use_container_width=True)
        
        # Delete user option
        st.subheader("🗑️ Remove User")
        name_to_delete = st.selectbox("Select User to Remove", [""] + list(unique_names.keys()))
        
        if name_to_delete and st.button("🗑️ Delete User"):
            confirm = st.checkbox("I confirm I want to delete this user")
            
            if confirm:
                with open(names_path, 'rb') as f:
                    names = pickle.load(f)
                with open(faces_path, 'rb') as f:
                    faces = pickle.load(f)
                
                # Get indices to keep
                indices_to_keep = [i for i, name in enumerate(names) if name != name_to_delete]
                
                # Filter data
                new_names = [names[i] for i in indices_to_keep]
                new_faces = faces[indices_to_keep]
                
                # Save filtered data
                with open(names_path, 'wb') as f:
                    pickle.dump(new_names, f)
                with open(faces_path, 'wb') as f:
                    pickle.dump(new_faces, f)
                
                st.success(f"✅ User {name_to_delete} removed successfully.")
                st.experimental_rerun()
    except Exception as e:
        st.error(f"❌ Error loading user data: {e}")

# ------------------- SETTINGS -------------------
def show_settings():
    st.subheader("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📧 Email Settings")
        email_sender = st.text_input("Email Sender", value=st.session_state.get('email_sender', ""))
        email_password = st.text_input("App Password", type="password")
        receiver = st.text_input("Default Receiver", value=st.session_state.get('email_receiver', ""))
        
        if st.button("💾 Save Email Settings"):
            st.warning("⚠️ For security, email settings are only saved for this session.")
            
            # Store in session state for the current session
            st.session_state['email_sender'] = email_sender
            st.session_state['email_password'] = email_password
            st.session_state['email_receiver'] = receiver
            
            st.success("✅ Email settings saved for this session.")
            st.info("ℹ️ In a production deployment, use environment variables or secrets.")
            
    with col2:
        st.subheader("🔐 Security Settings")
        
        # Admin credentials
        new_username = st.text_input("New Admin Username")
        new_password = st.text_input("New Admin Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("🔄 Change Admin Credentials"):
            if not new_username or not new_password:
                st.error("❌ Username and password cannot be empty.")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match.")
            else:
                # In a production app, store these securely
                st.session_state['admin_credentials'] = {new_username: new_password}
                st.success("✅ Admin credentials updated for this session.")
                st.info("ℹ️ In a production deployment, use environment variables or secrets.")
    
    # System information
    st.subheader("🖥️ System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Data Directory", len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]) if os.path.exists(DATA_DIR) else 0, "files")
    
    with col2:
        st.metric("Attendance Records", len([f for f in os.listdir(ATTENDANCE_DIR) if f.endswith('.csv')]) if os.path.exists(ATTENDANCE_DIR) else 0, "days")

    # Deployment information
    st.subheader("🚀 Deployment Information")
    st.info("""
    **To deploy this application:**
    
    1. Create a requirements.txt file with all dependencies
    2. Host on a platform like Streamlit Cloud, Heroku, or any cloud provider
    3. Set environment variables for email settings and credentials
    4. Configure webcam access for the deployment environment
    
    For webcam functionality in production, use streamlit-webrtc package.
    """)

# ------------------- MAIN APPLICATION -------------------
def main():
    # Page config
    st.set_page_config(
        page_title="Smart Face Recognition Attendance",
        page_icon="📸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
            .block-container {padding-top: 1rem;}
            .main .block-container {max-width: 1200px;}
            .stButton > button {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                border: none;
                padding: 0.5em 1em;
                font-weight: bold;
            }
            .stButton > button:hover {background-color: #45a049;}
            h1, h2, h3 {color: #1E88E5;}
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False
    
    # Title
    st.title("📸 Smart Face Recognition Attendance System")
    
    # Admin login page
    if not st.session_state['admin_logged_in']:
        # Login page
        st.header("🔐 Admin Login")
        
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown("### Sign In")
                username = st.text_input("👤 Username", placeholder="Enter your username")
                password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
                
                if st.button("🔓 Login"):
                    # Check credentials from session state first (if changed during session)
                    admin_credentials = st.session_state.get('admin_credentials', ADMIN_CREDENTIALS)
                    
                    if username in admin_credentials and admin_credentials[username] == password:
                        st.session_state['admin_logged_in'] = True
                        st.success("✅ Login successful!")
                        st.experimental_rerun()
                    else:
                        st.error("❌ Invalid credentials")
                
                st.info("ℹ️ Default login is username: 'admin', password: '1234'")
    else:
        # Create sidebar
        st.sidebar.title("📋 Navigation")
        
        # Menu options
        menu_options = {
            "Add Face": "➕ Add New Face",
            "Mark Attendance": "🧍 Take Attendance",
            "View Attendance": "📊 View Records",
            "Dashboard": "📈 Analytics Dashboard",
            "User Management": "👥 User Management",
            "Settings": "⚙️ Settings",
            "Logout": "🚪 Logout"
        }
        
        selection = st.sidebar.radio("Select Option", list(menu_options.keys()), format_func=lambda x: menu_options[x])
        
        # Display app info in sidebar
        with st.sidebar.expander("ℹ️ About"):
            st.write("Smart Face Recognition Attendance System")
            st.write("Version 1.0.0")
            st.write("Made with ❤️ by MGX")
        
        # Handle menu selection
        if selection == "Add Face":
            st.header("➕ Add New Face")
            
            name = st.text_input("👤 Enter Person's Name")
            
            if name:
                if st.button("📸 Start Face Capture"):
                    save_face(name)
            else:
                st.info("ℹ️ Please enter a name to continue.")
                
        elif selection == "Mark Attendance":
            st.header("🧍 Face Recognition Attendance")
            
            st.info("ℹ️ Click 'Start Attendance' to begin face recognition.")
            
            if st.button("▶️ Start Attendance"):
                mark_attendance()
                
        elif selection == "View Attendance":
            view_attendance()
            
        elif selection == "Dashboard":
            show_dashboard()
            
        elif selection == "User Management":
            user_management()
            
        elif selection == "Settings":
            show_settings()
            
        elif selection == "Logout":
            st.session_state['admin_logged_in'] = False
            st.experimental_rerun()

if __name__ == "__main__":
    main()
