import streamlit as st
import os
import time
from PIL import Image
import io
import numpy as np

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
    
    # Add an info box explaining the deployment status
    st.info("""
    🚧 This is a demo version of the Fashion Rating App. The full ML model pipeline requires 
    additional configuration on Streamlit Cloud. 
    
    For now, we're showcasing the UI and sample results. To see the full functionality, 
    you can run the app locally with all dependencies installed.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create a timestamp for the file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Instead of processing the image with our models, show demo results
        with st.spinner("Analyzing outfit..."):
            # Simulate processing delay
            time.sleep(2)
            
            # Display demo results
            with col2:
                st.subheader(f"Overall Score: 85/100")
                
                # Style Information
                st.write(f"**Style**: Fashion Forward Casual")
                st.write(f"**Fit**: Balanced Silhouette")
                
                # Color palette demo
                st.write("**Color Palette**:")
                palette_cols = st.columns(5)
                demo_colors = ["#000000", "#ffffff", "#0076a3", "#d3a121", "#7b3056"]
                color_names = ["Black", "White", "Deep Blue", "Gold", "Burgundy"]
                for i, (color, name) in enumerate(zip(demo_colors, color_names)):
                    palette_cols[i].markdown(
                        f"<div style='background-color:{color}; width:50px; height:20px;'></div>{name}",
                        unsafe_allow_html=True
                    )
            
            # Score Breakdown
            st.subheader("Score Breakdown")
            demo_scores = {
                "fit": 22, 
                "color": 21, 
                "footwear": 18,
                "accessories": 13, 
                "style": 20
            }
            component_names = {"fit": "Fit", "color": "Color", "footwear": "Footwear", 
                             "accessories": "Accessories", "style": "Style"}
            
            # Create progress bars for each component - using a safer approach
            for component, score in demo_scores.items():
                max_score = 25 if component in ["fit", "color"] else 20 if component == "footwear" else 15
                # Ensure percentage is between 0 and 1
                percentage = float(score) / float(max_score)
                percentage = max(0.0, min(1.0, percentage))  # Clamp between 0 and 1
                st.write(f"**{component_names.get(component, component.title())}**: {score}/{max_score}")
                st.progress(percentage)
            
            # Detailed Feedback
            st.subheader("Detailed Feedback")
            demo_feedback = {
                "fit": "The structured top creates a balanced silhouette with the relaxed bottom. The proportions work well for your frame, creating a confident, contemporary look.",
                "color": "The color combination shows sophisticated understanding of contrast. The neutral base with carefully selected accent colors creates a cohesive palette.",
                "footwear": "Your choice of footwear complements the outfit's style while adding a practical, grounded element. The proportions work well with the overall look.",
                "accessories": "The accessories enhance the outfit without overwhelming it. They create interest while maintaining a cohesive style direction.",
                "style": "You've successfully balanced casual elements with more refined pieces. The outfit demonstrates a clear understanding of current trends while maintaining personal style."
            }
            
            for component, text in demo_feedback.items():
                expander = st.expander(f"{component_names.get(component, component.title())} Feedback")
                expander.write(text)

if __name__ == "__main__":
    main()