import streamlit as st
import os
import asyncio
from PIL import Image
import io
import sys
import json
import time

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import our fashion rating pipeline
from app.core.pipeline import process_fashion_image

st.set_page_config(
    page_title="Fashion Outfit Rater",
    page_icon="👔",
    layout="wide"
)

def main():
    st.title("Fashion Outfit Rating App")
    st.write("""
    Upload an image of your outfit and get detailed fashion feedback and rating!
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create a temporary file to save the upload
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_file_path = f"temp_upload_{timestamp}.jpg"
        
        # Save uploaded file
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Display the uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Run analysis
        with st.spinner("Analyzing outfit..."):
            try:
                # Run the pipeline
                result = asyncio.run(process_fashion_image(temp_file_path))
                
                # Display results
                with col2:
                    st.subheader(f"Overall Score: {result['score']}/100")
                    
                    # Style Information
                    st.write(f"**Style**: {result['labels']['style'].replace('_', ' ').title()}")
                    st.write(f"**Fit**: {result['labels']['fit'].replace('_', ' ').title()}")
                    
                    # Color palette
                    if result['labels']['color_palette']:
                        st.write("**Color Palette**:")
                        palette_cols = st.columns(min(5, len(result['labels']['color_palette'])))
                        for i, color in enumerate(result['labels']['color_palette'][:5]):
                            color_name = color.replace('_', ' ').title()
                            if i < len(palette_cols):
                                palette_cols[i].markdown(
                                    f"<div style='background-color:{color}; width:50px; height:20px;'></div>{color_name}",
                                    unsafe_allow_html=True
                                )
                
                # Score Breakdown
                st.subheader("Score Breakdown")
                scores = result['score_breakdown']
                component_names = {"fit": "Fit", "color": "Color", "footwear": "Footwear", 
                                 "accessories": "Accessories", "style": "Style"}
                
                # Create progress bars for each component
                for component, score in scores.items():
                    # Determine max score for each component
                    max_score = 25 if component in ["fit", "color"] else 20 if component == "footwear" else 15
                    percentage = score / max_score
                    st.write(f"**{component_names.get(component, component.title())}**: {score}/{max_score}")
                    st.progress(percentage)
                
                # Detailed Feedback
                st.subheader("Detailed Feedback")
                feedback = result['feedback']
                for component, text in feedback.items():
                    expander = st.expander(f"{component_names.get(component, component.title())} Feedback")
                    expander.write(text)
                
                # Clean up the temp file
                try:
                    os.remove(temp_file_path)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"Error analyzing outfit: {str(e)}")
                # Clean up the temp file
                try:
                    os.remove(temp_file_path)
                except:
                    pass

if __name__ == "__main__":
    main()