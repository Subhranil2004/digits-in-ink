import streamlit as st
import tensorflow as tf
import time
from PIL import Image
import numpy as np

# import json
# import requests

# from streamlit_lottie import st_lottie

# Load the saved model
model = tf.keras.models.load_model(
    "./models/CNN_Augmented_100_model.h5", compile=False
)  # CNN_Augmented_100_model.h5


# Define a function for model inference
def predict(image):
    # Open and preprocess the image
    img = Image.open(image)
    img = img.resize((28, 28))  # Resize image to match model input size
    img = img.convert("L")
    img_array = np.array(img)  # Convert image to NumPy array
    img_array = np.expand_dims(img_array, axis=-1)  # Add batch dimension
    # st.write("Image shape:", img_array.shape)
    # Make predictions using the loaded model
    prediction = model.predict(np.expand_dims(img_array, axis=0))

    return prediction


# Streamlit app code
st.set_page_config(
    page_title="Classification of Handwritten Digits",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        # "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://github.com/Subhranil2004/handwritten-digit-classification/issues",
        # "About": "# This is a header. This is an *extremely* cool app!",
    },
)

# Sidebar
with st.sidebar:
    st.image(
        "./images/image.jpg",
        use_column_width=True,
        output_format="JPEG",
    )
st.sidebar.title("Classification of Handwritten Digits [0 - 9]")
st.sidebar.write(
    "The model is trained on the ***MNIST dataset*** and uses Convolutional Neural Network with Data augmentation. It has an exceptional accuracy rate of 99.45% on MNIST test dataset."
)

st.sidebar.write("\n")
st.sidebar.write(
    "There is always a scope for improvement and I would appreciate suggestions and/or constructive criticisms."
)


def navigate_to_github():
    """Hides an anchor tag and triggers a click on it to open the GitHub URL"""
    st.write(
        "<a href='https://github.com' target='_blank' style='display: none;'>Go to GitHub</a>",
        unsafe_allow_html=True,
    )


st.sidebar.link_button("GitHub", "https://github.com/Subhranil2004")

st.markdown(
    f"""
        <style>
            .sidebar {{
                width: 500px;
            }}
        </style>
    """,
    unsafe_allow_html=True,
)

# Main content

st.title("Classification of Handwritten Digits")
st.write("NOTE : Use pics with dark background for best results !")
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "png", "bmp", "tiff"],
)

if uploaded_file is not None:
    # Display the uploaded image with border
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width=300,
        clamp=True,
        # output_format="JPEG",
    )

    # Perform prediction
    if st.button("Predict"):
        result = predict(uploaded_file)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f"Progress: {i}%")
            time.sleep(0.00)

        status_text.text("Done!")

        # Display the prediction result
        max_index = np.argmax(result)

        st.write("Predicted Digit :  ", max_index, "  🎉")

        del_result = np.delete(result, max_index)
        res2 = np.argmax(del_result)
        if res2 >= max_index:
            res2 = res2 + 1

        expander = st.expander(
            ":orange[If you aren't satisfied with the result, check the prediction probabilities below ⬇⚠️]"
        )

        expander.write(f"Second most probable prediction: {res2}")
        expander.write(result)  # , res2, result
        # expander.write(result)

expander = st.expander("Some real life images to try with...", expanded=True)
expander.write("Just drag-and-drop your chosen image above ")
expander.image(
    [
        "./Real_Life_Images/three2.png",
        "./Real_Life_Images/six2.png",
        "./Real_Life_Images/two1.png",
        "./Real_Life_Images/nine1.png",
        "./Real_Life_Images/zero1.png",
        "./Real_Life_Images/five3.png",
        "./Real_Life_Images/eight3.png",
        # "./Real_Life_Images/one1.png",
    ],
    width=95,
)
expander.write(
    "All images might not give the desired result as the *1st* prediction due to low contrast. Check the probability scores in such cases."
)
expander = st.expander("View Model Training and Validation Results")
expander.write("Confusion Matrix: ")
expander.image("./images/CNN_ConfusionMatrix.png", use_column_width=True)
expander.write("Graphs: ")
expander.image("./images/CNN_Graphs.png", use_column_width=True)

expander = st.expander("If you are getting inaccurate results, follow these steps:")
expander.markdown(
    """    
    1. Use OneNote/MS Paint dark background    
    2. Write with well-contrast colours
    3. Thicken writing stroke
    4. Make sure digit occupies the maximum part of the image
    """
)
# Footer
st.write("\n\n\n")
st.markdown("---")
st.markdown(
    f"""Drop in any discrepancies or give suggestions in `Report a bug` option within the `⋮` menu"""
)

st.markdown(
    f"""<div style="text-align: right"> Developed by Subhranil Nandy </div>""",
    unsafe_allow_html=True,
)
