import streamlit as st
from backend import (
    generate_verification_pack,
    generate_verification_pack_from_text,
    suggest_attestation_chain,
    validate_document,
    validate_document_from_text,
    autofill_forms,
    generate_timeline,
    check_ocr_status
)
from utils import preprocess_image, validate_image_format, extract_profile_text
import io

st.set_page_config(page_title="EduVerify.AI", page_icon="🎓", layout="wide")
st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stTabs [data-baseweb="tab"] {font-size: 18px; font-weight: 600;}
    .stButton>button {background-color: #0052cc; color: white; border-radius: 6px;}
    .stTextInput>div>input {border-radius: 6px;}
</style>
""", unsafe_allow_html=True)

st.title("🎓 EduVerify.AI — GenAI Attestation Assistant")
st.write("AI-powered attestation, verification, and form autofill for degrees and marksheets.")

# Check OCR status and display
ocr_status = check_ocr_status()
if ocr_status["tesseract_available"]:
    st.success(f"🔍 {ocr_status['status_message']}")
else:
    st.warning(f"🔍 {ocr_status['status_message']}")
    st.info("💡 **Tip**: You can still use all features with manual text input!")

with st.sidebar:
    st.header("User Profile & Upload")
    user_role = st.selectbox("Select your role:", ["Doctor", "Engineer", "Teacher", "Student", "Other"])
    
    # File upload option
    uploaded_file = st.file_uploader("Upload scanned degree/marksheet (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image_bytes = uploaded_file.read()
        if not validate_image_format(image_bytes):
            st.error("Invalid image format. Please upload PNG or JPG.")
            image_bytes = None
        else:
            image_bytes = preprocess_image(image_bytes)
            if ocr_status["tesseract_available"]:
                st.success("✅ Image uploaded and processed successfully!")
            else:
                st.warning("⚠️ Image uploaded but OCR not available. Consider using manual text input.")
    else:
        image_bytes = None
    
    # Manual text input as backup
    st.subheader("Manual Text Input")
    if not ocr_status["tesseract_available"]:
        st.info("💡 OCR not available - manual input recommended!")
    st.write("Enter your degree details manually for accurate processing:")
    manual_text = st.text_area(
        "Enter degree details:",
        placeholder="e.g., Bachelor of Science in Computer Science, XYZ University, Graduated 2023, CGPA 3.8/4.0, Student: John Doe",
        height=120
    )
    
    use_manual_input = st.checkbox("Use manual text instead of image")
    
    st.write("Your role and degree information will be used for all features.")

# Tabs for features
tabs = st.tabs([
    "AI-Generated Verification Pack",
    "Institution Chain Matcher",
    "Document Validator Bot",
    "Form Autofiller",
    "Track & Next-Steps Timeline"
])

# 1. Verification Pack
with tabs[0]:
    st.subheader("AI-Generated Verification Pack")
    st.write("Generate attestation letters, embassy drafts, MOFA forms, and a checklist.")
    if st.button("Generate Verification Pack"):
        if use_manual_input and manual_text.strip() and user_role:
            with st.spinner("Generating attestation documents from text input..."):
                result = generate_verification_pack_from_text(manual_text, user_role)
            for k, v in result.items():
                st.markdown(f"### {k.replace('_', ' ').title()}")
                st.text_area(f"{k}_content", value=v, height=200, key=f"display_{k}", disabled=True)
                st.download_button(f"Download {k.replace('_', ' ').title()}", v, file_name=f"{k}.txt")
                st.markdown("---")
        elif image_bytes and user_role:
            with st.spinner("Generating attestation documents from image..."):
                result = generate_verification_pack(image_bytes, user_role)
            for k, v in result.items():
                st.markdown(f"### {k.replace('_', ' ').title()}")
                st.text_area(f"{k}_content", value=v, height=200, key=f"display_{k}", disabled=True)
                st.download_button(f"Download {k.replace('_', ' ').title()}", v, file_name=f"{k}.txt")
                st.markdown("---")
        else:
            st.warning("Please either upload a valid scan or enter manual text, and select your role.")

# 2. Institution Chain Matcher
with tabs[1]:
    st.subheader("Institution Chain Matcher")
    st.write("AI suggests the exact attestation chain based on your profile.")
    if st.button("Suggest Attestation Chain"):
        if use_manual_input and manual_text.strip():
            with st.spinner("Analyzing profile and matching chain..."):
                chain = suggest_attestation_chain(manual_text)
            st.markdown("**Attestation Chain:**")
            st.write(" → ".join(chain))
        elif image_bytes:
            with st.spinner("Extracting profile and matching chain..."):
                profile_text = extract_profile_text(image_bytes)
                chain = suggest_attestation_chain(profile_text)
            st.markdown("**Attestation Chain:**")
            st.write(" → ".join(chain))
        else:
            st.warning("Please either upload a valid scan or enter manual text.")

# 3. Document Validator Bot
with tabs[2]:
    st.subheader("Document Validator Bot")
    st.write("Checks if your document is ready for attestation and flags issues.")
    if st.button("Validate Document"):
        if use_manual_input and manual_text.strip():
            with st.spinner("Validating document information..."):
                result = validate_document_from_text(manual_text)
            st.markdown(f"**Ready for Attestation:** {'✅' if result['ready'] else '❌'}")
            st.markdown("**Issues:**")
            for issue in result.get("issues", []):
                st.write(f"- {issue}")
            st.markdown("**Information Provided:**")
            st.json(result.get("fields", {}))
        elif image_bytes:
            with st.spinner("Validating document..."):
                result = validate_document(image_bytes)
            st.markdown(f"**Ready for Attestation:** {'✅' if result['ready'] else '❌'}")
            st.markdown("**Issues:**")
            for issue in result.get("issues", []):
                st.write(f"- {issue}")
            st.markdown("**Fields Extracted:**")
            st.json(result.get("fields", {}))
        else:
            st.warning("Please either upload a valid scan or enter manual text.")

# 4. Form Autofiller
with tabs[3]:
    st.subheader("Form Autofiller")
    st.write("Auto-fills MOFA, HEC, WES, Embassy forms from your profile.")
    if st.button("Auto-fill Forms"):
        if use_manual_input and manual_text.strip():
            with st.spinner("Auto-filling forms from text input..."):
                forms = autofill_forms(manual_text)
            for k, v in forms.items():
                st.markdown(f"### {k.replace('_', ' ').title()}")
                st.text_area(f"{k}_form_content", value=v, height=200, key=f"form_{k}", disabled=True)
                st.download_button(f"Download {k.replace('_', ' ').title()}", v, file_name=f"{k}.txt")
                st.markdown("---")
        elif image_bytes:
            with st.spinner("Extracting profile and auto-filling forms..."):
                profile_text = extract_profile_text(image_bytes)
                forms = autofill_forms(profile_text)
            for k, v in forms.items():
                st.markdown(f"### {k.replace('_', ' ').title()}")
                st.text_area(f"{k}_form_content", value=v, height=200, key=f"form_{k}", disabled=True)
                st.download_button(f"Download {k.replace('_', ' ').title()}", v, file_name=f"{k}.txt")
                st.markdown("---")
        else:
            st.warning("Please either upload a valid scan or enter manual text.")

# 5. Track & Next-Steps Timeline
with tabs[4]:
    st.subheader("Track & Next-Steps Timeline")
    st.write("Generates a timeline with deadlines and recommended actions.")
    if st.button("Generate Timeline"):
        if use_manual_input and manual_text.strip():
            with st.spinner("Generating timeline from text input..."):
                timeline = generate_timeline(manual_text)
            st.markdown("### Attestation Timeline & Actions")
            st.text_area("timeline_content", value=timeline, height=300, key="timeline_display", disabled=True)
            st.download_button("Download Timeline", timeline, file_name="timeline.txt")
        elif image_bytes:
            with st.spinner("Extracting profile and generating timeline..."):
                profile_text = extract_profile_text(image_bytes)
                timeline = generate_timeline(profile_text)
            st.markdown("### Attestation Timeline & Actions")
            st.text_area("timeline_content", value=timeline, height=300, key="timeline_display", disabled=True)
            st.download_button("Download Timeline", timeline, file_name="timeline.txt")
        else:
            st.warning("Please either upload a valid scan or enter manual text.")

st.markdown("---")
st.markdown("**EduVerify.AI** — Powered by Groq GenAI + Tesseract OCR. All data processed locally.")
