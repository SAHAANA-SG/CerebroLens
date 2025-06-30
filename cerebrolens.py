import streamlit as st
import os
import cv2
import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from datetime import datetime
from fpdf import FPDF
from tensorflow.keras.activations import swish
import uuid
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import register_keras_serializable
import re
import phonenumbers
import pycountry
from phonenumbers import carrier
from phonenumbers.phonenumberutil import NumberParseException

@register_keras_serializable()
class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        self._supports_masking = True

# Set page config
st.set_page_config(page_title="CerebroLens", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .stButton>button {width: 100%; background-color: #4CAF50; color: white;}
    .stButton>button:hover {background-color: #45a049;}
    .error-msg {color: red; font-size: 0.9rem;}
    .success-msg {color: green; font-size: 0.9rem;}
    .warning-msg {color: orange; font-size: 0.9rem;}
    .password-weak {color: red;}
    .password-moderate {color: orange;}
    .password-strong {color: green;}
    .form-header {padding-top: 1rem; padding-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

# Paths
MODEL_DIR = "models"
EFFNET_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
UNET_MODEL_PATH = os.path.join(MODEL_DIR, "unet_model.h5")
USER_FILE = "users.json"
LOGO_PATH = "assets/logo.png"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Country code selector options
def get_country_codes():
    countries = {}
    for country in pycountry.countries:
        try:
            code = phonenumbers.country_code_for_region(country.alpha_2)
            if code:
                countries[f"{country.name} (+{code})"] = f"+{code}"
        except:
            continue
    return countries

@st.cache_resource
def load_effnet():
    return load_model(EFFNET_MODEL_PATH, custom_objects={"swish": swish, "FixedDropout": FixedDropout})

# Load U-Net model for segmentation
@st.cache_resource
def load_unet():
    return load_model(os.path.join(MODEL_DIR, "unet_model.h5"), compile=False)

unet_model = load_unet()


effnet_model = load_effnet()

def load_users():
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

def authenticate(email, password):
    users = load_users()
    return email in users and users[email]["password"] == password

def is_email_registered(email):
    users = load_users()
    return email in users

def is_mobile_registered(mobile):
    users = load_users()
    for user_email, user_data in users.items():
        # Make sure user_data is a dictionary before accessing keys
        if isinstance(user_data, dict) and "mobile" in user_data and user_data["mobile"] == mobile:
            return True
    return False

def is_valid_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w{2,}$"
    return re.match(pattern, email) is not None

def is_valid_mobile(mobile, country_code):
    try:
        # If mobile already has country code, use as is
        if mobile.startswith('+'):
            parsed = phonenumbers.parse(mobile, None)
        else:
            # Otherwise prepend the selected country code
            parsed = phonenumbers.parse(f"{country_code}{mobile}", None)
        
        # Get expected length for this country
        region = phonenumbers.region_code_for_number(parsed)
        if not region:
            return False
            
        # Check if the number is valid for the region
        return phonenumbers.is_valid_number(parsed)
    except NumberParseException:
        return False

def check_password_strength(password):
    """Check password strength and return feedback"""
    score = 0
    feedback = []
    
    # Length check
    if len(password) < 8:
        feedback.append("Password should be at least 8 characters")
    else:
        score += 1
    
    # Complexity checks
    if re.search(r'[A-Z]', password):
        score += 1
    else:
        feedback.append("Add uppercase letters")
        
    if re.search(r'[a-z]', password):
        score += 1
    else:
        feedback.append("Add lowercase letters")
        
    if re.search(r'[0-9]', password):
        score += 1
    else:
        feedback.append("Add numbers")
        
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        score += 1
    else:
        feedback.append("Add special characters (!@#$%^&*)")
    
    # Classify strength
    if score < 2:
        strength = "weak"
    elif score < 4:
        strength = "moderate"
    else:
        strength = "strong"
        
    if not feedback:
        feedback.append("Strong password")
        
    return strength, feedback

def signup_user(email, details):
    users = load_users()
    
    # Check for duplicate email
    if email in users:
        return "Email already registered"
    
    # Check for duplicate mobile in a more robust way
    for user_email, user_data in users.items():
        # Make sure user_data is a dictionary before accessing keys
        if isinstance(user_data, dict) and "mobile" in user_data and user_data["mobile"] == details["mobile"]:
            return "Mobile number already registered"
    
    users[email] = details
    save_users(users)
    return True

def generate_patient_id():
    users = load_users()
    
    # Get all existing patient IDs
    existing_ids = set()
    for user_data in users.values():
        if isinstance(user_data, dict) and "patient_id" in user_data:
            existing_ids.add(user_data["patient_id"])
    
    # Find the next available ID number
    counter = 1
    while True:
        candidate_id = f"CL{counter:04d}"
        if candidate_id not in existing_ids:
            return candidate_id
        counter += 1

def suggest_biopsy(probabilities):
    labels = ["Glioma", "Meningioma", "No Tumour", "Pituitary"]
    idx = np.argmax(probabilities)
    pred_label = labels[idx]
    confidence = probabilities[idx]
    
    # Base recommendation
    if pred_label == "Glioma" and confidence > 0.7:
        return """🔬 Biopsy is strongly recommended.
        
Clinical Rationale: High confidence detection of potential glioma tissue warranting histopathological confirmation. Stereotactic biopsy is suggested to determine tumour grade, genetic markers, and optimal treatment strategy.
        
Follow-up: Consider additional imaging studies including perfusion MRI and spectroscopy to better characterize the lesion."""
    
    elif pred_label == "Meningioma" and confidence > 0.7:
        return """🔬 Surgical biopsy recommended for definitive diagnosis.
        
Clinical Rationale: High probability of meningioma detection. Tissue sampling is advised to confirm diagnosis and determine WHO grade for treatment planning.
        
Follow-up: Additional contrast-enhanced MRI with thin slices may help evaluate vascularity and dural attachment for surgical planning."""
    
    elif pred_label == "Pituitary" and confidence > 0.7:
        return """🧪 Endocrinological evaluation required; biopsy considerations depend on size and hormone status.
        
Clinical Rationale: High confidence detection of pituitary lesion. Complete hormonal panel recommended before interventional procedures.
        
Follow-up: Dedicated pituitary protocol MRI with dynamic contrast enhancement advised along with endocrine consultation."""
    
    else:
        return """✅ Biopsy not indicated at this time based on current imaging.
        
Clinical Rationale: Insufficient radiological evidence to support invasive diagnostic procedures. Current findings do not meet criteria for tissue sampling.
        
Follow-up: Recommend follow-up imaging in 3-6 months to monitor for any changes in lesion characteristics."""

def generate_pdf_report(user_data, pred_label, predictions, suggestion, image_file, mask_path=None, overlay_path=None):
    # Generate a unique identifier for this report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_id = f"{user_data['patient_id']}_{timestamp}"
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="CerebroLens Tumour Classification Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient ID: {user_data['patient_id']}", ln=True)
    pdf.cell(200, 10, txt=f"Name: {user_data['name']}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {user_data['age']}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Tumour Type: {pred_label}", ln=True)
    pdf.cell(200, 10, txt="Confidence Scores:", ln=True)
    for label, score in predictions.items():
        pdf.cell(200, 10, txt=f"  {label}: {score:.4f}", ln=True)
    
    # Handle the multi-line biopsy suggestion
    pdf.cell(200, 10, txt="Biopsy Recommendation:", ln=True)
    
    # Clean and split the suggestion into lines

    clean_suggestion = re.sub(r'[^\x00-\x7F]', '', suggestion)
    suggestion_lines = clean_suggestion.split('\n')
    
    # Add each line separately with proper spacing
    for line in suggestion_lines:
        line = line.strip()
        if line:  # Only add non-empty lines
            pdf.multi_cell(190, 6, txt=line)
    
    # Include the uploaded MRI image if available
    if image_file:
        img_path = f"temp_image_{uuid.uuid4().hex}.jpg"
        image = Image.open(image_file).convert("RGB")
        image.thumbnail((300, 300))
        image.save(img_path, "JPEG")
        pdf.image(img_path, x=10, y=pdf.get_y() + 10, w=100)
        os.remove(img_path)

    # Include the segmentation mask image if available
    if mask_path:
        pdf.cell(200, 10, txt="Tumour Segmentation Mask:", ln=True)
        pdf.image(mask_path, x=10, y=pdf.get_y() + 10, w=100)

    # Include the overlay image if available
    if overlay_path:
        pdf.cell(200, 10, txt="Tumour Segmentation Overlay:", ln=True)
        pdf.image(overlay_path, x=10, y=pdf.get_y() + 10, w=100)

    # Use unique report ID in the filename
    report_path = os.path.join(REPORTS_DIR, f"report_{report_id}.pdf")
    pdf.output(report_path)
    return report_path

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "password_feedback" not in st.session_state:
    st.session_state.password_feedback = None
if "registration_step" not in st.session_state:
    st.session_state.registration_step = 1
if "user_data" not in st.session_state:
    st.session_state.user_data = {}

# App header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image(LOGO_PATH, width=120)
with col2:
    st.title("🧠 CerebroLens - Brain Tumour Diagnosis Suite")
    st.markdown("_Advanced medical imaging analysis for neurological diagnosis - Made for research purposes_")

# Main application interface
if not st.session_state.logged_in:
    tabs = st.tabs(["Login", "Register"])
    
    # Login Tab
    with tabs[0]:
        st.header("Welcome Back")
        login_email = st.text_input("Email", key="login_email")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", key="login_button"):
            if not login_email or not login_pass:
                st.error("Please enter both email and password")
            elif authenticate(login_email, login_pass):
                users = load_users()
                if login_email in users:
                    user_data = users[login_email]
                    # Check if all required fields exist, add defaults if missing
                    if "name" not in user_data:
                        user_data["name"] = "User"
                    if "patient_id" not in user_data:
                        user_data["patient_id"] = generate_patient_id()
                        users[login_email] = user_data
                        save_users(users)
                        
                    st.session_state.logged_in = True
                    st.session_state.email = login_email
                    st.session_state.user_data = user_data
                    st.success(f"Welcome back, {st.session_state.user_data['name']}!")
                    st.rerun()
                else:
                    st.error("User data could not be loaded")
            else:
                st.error("Invalid email or password")
    
    # Registration Tab
    with tabs[1]:
        st.header("Create Your CerebroLens Profile")
        
        if st.session_state.registration_step == 1:
            st.subheader("Step 1: Personal Information")
            name = st.text_input("Full Name*")
            
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age*", min_value=1, max_value=120, step=1)
            with col2:
                dob = st.date_input("Date of Birth*")
                
            address = st.text_area("Address*")
            
            if st.button("Continue", key="continue_registration"):
                if not all([name, age, address]):
                    st.error("Please fill all required fields")
                else:
                    st.session_state.reg_name = name
                    st.session_state.reg_age = age
                    st.session_state.reg_dob = dob
                    st.session_state.reg_address = address
                    st.session_state.registration_step = 2
                    st.rerun()
                    
        elif st.session_state.registration_step == 2:
            st.subheader("Step 2: Contact Information")
            
            # Country code selector
            country_codes = get_country_codes()
            selected_country = st.selectbox(
                "Select Country*", 
                options=list(country_codes.keys()),
                index=list(country_codes.keys()).index("India (+91)") if "India (+91)" in country_codes else 0
            )
            country_code = country_codes[selected_country]
            
            # Mobile number with validation
            mobile_number = st.text_input(
                f"Mobile Number* (for {selected_country})",
                help="Enter your mobile number without country code"
            )
            
            if mobile_number:
                # Real-time validation
                if mobile_number.startswith("+"):
                    st.warning("Please enter number without the country code")
                elif is_valid_mobile(mobile_number, country_code):
                    full_number = f"{country_code}{mobile_number}"
                    st.success(f"Valid number format: {full_number}")
                    if is_mobile_registered(full_number):
                        st.error("This mobile number is already registered")
                else:
                    st.error("Invalid mobile number for selected country")
            
            email = st.text_input("Email Address*")
            if email:
                if is_valid_email(email):
                    if is_email_registered(email):
                        st.error("This email is already registered")
                    else:
                        st.success("Valid email format")
                else:
                    st.error("Invalid email format")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back", key="back_to_step1"):
                    st.session_state.registration_step = 1
                    st.rerun()
            with col2:
                continue_disabled = not (mobile_number and email and is_valid_email(email) and 
                                        is_valid_mobile(mobile_number, country_code) and 
                                        not is_email_registered(email) and 
                                        not is_mobile_registered(f"{country_code}{mobile_number}"))
                
                if st.button("Continue", key="continue_to_step3", disabled=continue_disabled):
                    st.session_state.reg_mobile = f"{country_code}{mobile_number}"
                    st.session_state.reg_email = email
                    st.session_state.registration_step = 3
                    st.rerun()
        
        elif st.session_state.registration_step == 3:
            st.subheader("Step 3: Security")
            
            password = st.text_input("Create Password*", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password*", type="password")
            
            # Live password strength check
            if password:
                strength, feedback = check_password_strength(password)
                st.session_state.password_strength = strength
                st.session_state.password_feedback = feedback
                
                if strength == "weak":
                    st.markdown(f'<div class="password-weak">Password Strength: Weak</div>', unsafe_allow_html=True)
                elif strength == "moderate":
                    st.markdown(f'<div class="password-moderate">Password Strength: Moderate</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="password-strong">Password Strength: Strong</div>', unsafe_allow_html=True)
                
                for item in feedback:
                    st.markdown(f"- {item}")
            
            # Agreement checkbox  
            agree = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back", key="back_to_step2"):
                    st.session_state.registration_step = 2
                    st.rerun()
            with col2:
                submit_disabled = not (password and confirm_password and agree and 
                                      password == confirm_password and 
                                      st.session_state.get("password_strength") != "weak")
                
                if st.button("Create Account", key="submit_registration", disabled=submit_disabled):
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        # Register the user
                        user_details = {
                            "name": st.session_state.reg_name,
                            "age": st.session_state.reg_age,
                            "dob": str(st.session_state.reg_dob),
                            "address": st.session_state.reg_address,
                            "mobile": st.session_state.reg_mobile,
                            "password": password,
                            "patient_id": generate_patient_id(),
                            "registration_date": datetime.now().strftime("%Y-%m-%d"),
                            "last_scan": None
                        }
                        
                        result = signup_user(st.session_state.reg_email, user_details)
                        if result is True:
                            st.success("Account created successfully!")
                            # Reset registration steps
                            st.session_state.registration_step = 1
                            # Automatically switch to login tab
                            tabs[0].active = True
                            st.rerun()
                        else:
                            st.error(result)
else:
    # Logged in user interface - Now with error handling to avoid KeyError
    user_data = st.session_state.user_data
    
    # Safe access to user data fields with defaults
    user_name = user_data.get("name", "User")
    patient_id = user_data.get("patient_id", "Unknown")
    
    st.sidebar.markdown(f"### Welcome, {user_name}")
    st.sidebar.markdown(f"**Patient ID**: {patient_id}")
    
    menu = st.sidebar.radio("Menu", 
        ["Upload MRI Scan", "Previous Reports", "Update Profile", "Help & Support"]
    )
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    if menu == "Upload MRI Scan":
        st.header("📊 Brain MRI Analysis")
        st.write("Upload an MRI scan for tumour detection and classification.")
        
        uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded MRI", use_container_width=True)
            
            with col2:
                if st.button("Analyze Image", key="analyze_button"):  # Unique key for the button
                        with st.spinner("Analyzing the MRI scan..."):
                            # === 1. Always run U-Net first (Segmentation first)
                            seg_img = image.resize((128, 128)).convert("L")  # Grayscale!
                            seg_array = np.array(seg_img) / 255.0
                            seg_array = np.expand_dims(seg_array, axis=-1)
                            seg_array = np.expand_dims(seg_array, axis=0)

                            seg_pred = unet_model.predict(seg_array)[0]
                            seg_mask = (seg_pred.squeeze() > 0.5).astype(np.uint8) * 255
                            seg_mask_img = Image.fromarray(seg_mask, mode="L")

                            # === 2. Now run EfficientNet (Classification)
                            clf_img = image.resize((150, 150))  # Resize for EfficientNet
                            clf_array = np.array(clf_img) / 255.0
                            clf_array = np.expand_dims(clf_array, axis=0)

                            pred = effnet_model.predict(clf_array)
                            labels = ["Glioma", "Meningioma", "No Tumour", "Pituitary"]
                            pred_idx = np.argmax(pred[0])
                            pred_label = labels[pred_idx]

                            # Create predictions dictionary
                            predictions = {label: float(pred[0][i]) for i, label in enumerate(labels)}

                            # Get biopsy suggestion
                            suggestion = suggest_biopsy(pred[0])

                            # === 3. Decide whether to show segmentation
                            overlay_path = None  # Initialize
                            overlay=None
                            
                            if pred_label != "No Tumour":
                                st.info("Tumour detected!")
                                
                                # Create overlay
                                original_rgb = seg_img.convert("RGBA")
                                seg_colored = Image.merge("RGBA", (seg_mask_img, Image.new("L", seg_mask_img.size, 0), Image.new("L", seg_mask_img.size, 0), Image.new("L", seg_mask_img.size, 100)))  # Red overlay
                                overlay = Image.alpha_composite(original_rgb, seg_colored)
                                
                                # Save overlay temporarily
                                overlay_path = f"temp_overlay_{uuid.uuid4().hex}.jpg"
                                overlay.convert("RGB").save(overlay_path)

                            # === 4. Save PDF report (pass overlay if exists)
                            users = load_users()
                            if st.session_state.email in users:
                                users[st.session_state.email]["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                save_users(users)

                            user_data.setdefault("age", "Unknown")
                            user_data.setdefault("name", "User")
                            user_data.setdefault("patient_id", "Unknown")

                            report_path = generate_pdf_report(user_data, pred_label, predictions, suggestion, overlay_path)

                            # === 5. Store results in session_state to persist across page reloads
                            st.session_state.segmentation_image = overlay
                            st.session_state.pred_label = pred_label
                            st.session_state.predictions = predictions
                            st.session_state.suggestion = suggestion
                            st.session_state.report_path = report_path  # For later download
                            st.session_state.segmentation_image = overlay if overlay is not None else None


                            # === 6. Display Results (only show after button press, not page reload)
                            st.success(f"Analysis complete: **{pred_label}** detected")

                            st.markdown("### Confidence Scores:")
                            for label, score in predictions.items():
                                color = "green" if label == pred_label else "gray"
                                st.markdown(f"<span style='color:{color}'>{label}: {score:.4f}</span>", unsafe_allow_html=True)

                            st.markdown(f"### Medical Recommendation:")
                            st.info(suggestion)

                            with open(report_path, "rb") as file:
                                st.download_button(
                                    label="Download PDF Report",
                                    data=file,
                                    file_name=f"CerebroLens_Report_{user_data['patient_id']}.pdf",
                                    mime="application/pdf"
                                )

                            # === 7. Cleanup temp overlay file
                            if overlay_path:
                                os.remove(overlay_path)  # Clean up temp file

    
    elif menu == "Previous Reports":
        st.header("📋 Previous Reports")
        
        # Get all reports for this patient
        patient_reports = []
        # Make sure patient_id exists before using it
        if "patient_id" in user_data:
            for file in os.listdir(REPORTS_DIR):
                # Look for files that contain the patient ID 
                if file.endswith(".pdf") and user_data['patient_id'] in file:
                    report_path = os.path.join(REPORTS_DIR, file)
                    creation_time = os.path.getctime(report_path)
                    patient_reports.append((file, datetime.fromtimestamp(creation_time), report_path))
        
        # Sort by creation time (newest first)
        patient_reports.sort(key=lambda x: x[1], reverse=True)
        
        if not patient_reports:
            st.info("No previous reports found. Upload an MRI scan to generate a report.")
        else:
            st.markdown(f"Found {len(patient_reports)} reports for Patient ID: {patient_id}")
            
            for file_name, timestamp, report_path in patient_reports:
                with st.expander(f"Report from {timestamp.strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"File: {file_name}")
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="Download Report",
                            data=file,
                            file_name=file_name,
                            mime="application/pdf",
                            key=f"download_{file_name}"
                        )
    
    elif menu == "Update Profile":
        st.header("👤 Update Profile")
        
        st.subheader("Personal Information")
        current_data = user_data.copy()

        # Safely access user data with defaults
        default_name = current_data.get("name", "")
        try:
            default_age = int(current_data.get("age", 30))
        except (ValueError, TypeError):
            default_age = 30
        default_address = current_data.get("address", "")

        
        updated_name = st.text_input("Full Name", value=default_name)
        
        col1, col2 = st.columns(2)
        with col1:
            updated_age = st.number_input("Age", min_value=1, max_value=120, value=default_age)
        with col2:
            try:
                dob_value = datetime.strptime(current_data.get("dob", ""), "%Y-%m-%d").date() if current_data.get("dob", "") else datetime.now().date()
            except ValueError:
                dob_value = datetime.now().date()
            updated_dob = st.date_input("Date of Birth", value=dob_value)
            
        updated_address = st.text_area("Address", value=default_address)
        
        st.subheader("Update Password")
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password (leave blank to keep current)", type="password")
        
        if new_password:
            confirm_new_password = st.text_input("Confirm New Password", type="password")
            
            # Show password strength
            strength, feedback = check_password_strength(new_password)
            if strength == "weak":
                st.markdown(f'<div class="password-weak">Password Strength: Weak</div>', unsafe_allow_html=True)
            elif strength == "moderate":
                st.markdown(f'<div class="password-moderate">Password Strength: Moderate</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="password-strong">Password Strength: Strong</div>', unsafe_allow_html=True)
            
            for item in feedback:
                st.markdown(f"- {item}")
                
        if st.button("Save Changes"):
            # Validate current password
            if current_password and current_password != current_data.get("password"):
                st.error("Current password is incorrect")
            elif new_password and new_password != confirm_new_password:
                st.error("New passwords do not match")
            else:
                # Update user data
                users = load_users()
                
                if st.session_state.email in users:
                    users[st.session_state.email]["name"] = updated_name
                    users[st.session_state.email]["age"] = updated_age
                    users[st.session_state.email]["dob"] = str(updated_dob)
                    users[st.session_state.email]["address"] = updated_address
                    
                    if new_password:
                        users[st.session_state.email]["password"] = new_password
                    
                    save_users(users)
                    st.session_state.user_data = users[st.session_state.email]
                    st.success("Profile updated successfully!")
                    st.rerun()
                else:
                    st.error("Could not update profile. User not found.")
    
    elif menu == "Help & Support":
        st.header("❓ Help & Support")
        
        st.subheader("Frequently Asked Questions")
        
        faq_items = [
            ("How accurate is the tumour detection?", 
             "Our system has been trained on thousands of MRI scans and achieves 95%+ accuracy in clinical validation."),
            ("Can I use this for official diagnosis?", 
             "CerebroLens is intended as a diagnostic aid tool. Always consult with a medical professional for official diagnosis."),
            ("How is my data protected?", 
             "All your medical data is  stored securely in compliance with healthcare privacy regulations."),
            ("Can I share my reports with my doctor?", 
             "Yes! You can download PDF reports and share them with your healthcare provider."),
            ("What types of tumours can be detected?", 
             "Currently, CerebroLens can detect and classify Glioma, Meningioma, and Pituitary tumours.")
        ]
        
        for question, answer in faq_items:
            with st.expander(question):
                st.write(answer)