import streamlit as st
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure the generative AI model
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Update the system prompt
system_prompt = """
As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining medical images for a renowned hospital. Your expertise is crucial in identifying any anomalies, diseases, or health issues that may be present in the image.

Your Responsibilities Include:

1. Detailed Analysis: Thoroughly analyze each image focusing on identifying any abnormal findings.
2. Finding report: Document all observed anomalies or signs of diseases. Clearly articulate these findings in a structured format.
3. Recommendation and next step: Based on your analysis, suggest potential next steps including further tests or treatments as applicable.
4. Treatment and suggestion: If appropriate, recommend possible treatment options or interventions.
5. First Aid Tips: Provide some first aid medical tips relevant to the image.

Important notes:

1. Scope of Response: Only respond if the image pertains to human health issues.
2. Clarity of Image: In cases where the image quality impedes clear analysis, note that certain aspects are unable to be determined based on the provided image.
3. Disclaimer: Accompany your analysis with a disclaimer; consult a doctor before making any decisions.
4. Your insights are valuable in guiding clinical decisions. Please proceed with the analysis adhering to the structured approach outlined above.

Response:
"""

# Initialize Streamlit app
st.set_page_config(page_title="Medical Image Analysis", page_icon=":microscope:")

st.title("Medical Image Analysis")
st.subheader("Upload an image to get insights")

# Display a placeholder image
placeholder_image_path = Path("path/to/your/placeholder_image.png")
if placeholder_image_path.exists():
    st.image(str(placeholder_image_path), caption="Placeholder Image", use_column_width=True)

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# Additional user input fields
st.subheader("Provide additional details for first aid suggestions")
symptoms = st.text_area("Describe any symptoms or relevant information")
duration = st.text_input("How long have you had this problem?")
severity = st.radio("How severe is the problem?", ("Mild", "Moderate", "Severe"))
allergies = st.text_area("Do you have any known allergies?")

# Submit button
submit_button = st.button("Generate Insights")

# Model generation function
def generate_insights(image_data, symptoms, duration, severity, allergies):
    image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
    prompt_parts = [
        image_parts[0], 
        system_prompt + f"""
        Additional details:
        - Symptoms: {symptoms}
        - Duration: {duration}
        - Severity: {severity}
        - Allergies: {allergies}
        """
    ]
    response = genai.GenerativeModel(
        model_name="gemini-pro-vision",
        generation_config=generation_config,
        safety_settings=safety_settings,
    ).generate_content(prompt_parts)
    return response.text

# Handle the form submission
if submit_button:
    if uploaded_file is not None:
        with st.spinner("Analyzing image..."):
            image_data = uploaded_file.read()
            try:
                response_text = generate_insights(image_data, symptoms, duration, severity, allergies)
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                
                # Parse response to extract first aid tips and insights
                response_lines = response_text.split('\n')
                insights = []
                first_aid_tips = []

                for line in response_lines:
                    if "First Aid Tips:" in line:
                        first_aid_index = response_lines.index(line) + 1
                        first_aid_tips = response_lines[first_aid_index:]
                    else:
                        insights.append(line)
                
                st.write("\n".join(insights))
                
                # Adding first aid tips
                if first_aid_tips:
                    st.subheader("First Aid Tips")
                    st.write("Based on the analysis, here are some first aid tips:")
                    for tip in first_aid_tips:
                        st.write(tip)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload an image before submitting.")

# Footer disclaimer
st.markdown(
    """
    **Disclaimer:** This tool is intended for informational purposes only. 
    Consult a healthcare professional before making any medical decisions.
    """
)
