import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import base64

from io import BytesIO
from datetime import datetime

from PIL import Image
from torchvision import models


# ==================================================
# STEP 1: PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Building Defect AI",
    page_icon="🏢",
    layout="wide"
)


# ==================================================
# STEP 2: CUSTOM STYLING
# ==================================================

st.markdown("""
<style>

.main-title {
    font-size: 34px;
    font-weight: 700;
    color: #183B56;
}

.subtitle {
    color: #718096;
    font-size: 16px;
}

.result-card {
    background-color: #F0F5F9;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}

.result-label {
    color: #718096;
    font-size: 14px;
}

.result-value {
    color: #183B56;
    font-size: 25px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# STEP 3: LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        "models/building_defect_resnet18_experiment3.pth",
        map_location=device,
        weights_only=True
    )

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        checkpoint["num_classes"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return (
        model,
        device,
        checkpoint["class_names"],
        weights.transforms()
    )


model, device, class_names, transform = load_model()


# ==================================================
# STEP 4: GRAD-CAM FUNCTION
# ==================================================

def generate_gradcam(input_tensor, target_class):

    activations = {}
    gradients = {}

    # Capture feature maps and their gradients

    def forward_hook(module, inputs, output):

        activations["features"] = output

        output.register_hook(
            lambda grad: gradients.update(
                {"gradients": grad}
            )
        )

    target_layer = model.layer4[-1].conv2

    hook = target_layer.register_forward_hook(
        forward_hook
    )

    try:

        model.zero_grad()

        # Forward pass

        with torch.enable_grad():

            outputs = model(input_tensor)

            score = outputs[0, target_class]

            # Calculate gradients

            score.backward()

        feature_maps = activations["features"]

        feature_gradients = gradients["gradients"]

        # Calculate feature importance

        importance_weights = feature_gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Combine feature maps

        cam = (
            importance_weights * feature_maps
        ).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        # Resize heatmap

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        # Normalise heatmap

        cam = cam - cam.min()

        cam = cam / (cam.max() + 1e-8)

        return cam

    finally:

        hook.remove()


# ==================================================
# STEP 5: CREATE HEATMAP AND OVERLAY
# ==================================================

def create_overlay(image, cam):

    display_image = image.resize((224, 224))

    image_array = np.array(display_image)

    # Convert heatmap into colours

    heatmap_colours = plt.get_cmap("jet")(
        cam
    )[:, :, :3]

    heatmap_colours = (
        heatmap_colours * 255
    ).astype(np.uint8)

    # Blend original image and heatmap

    overlay = (
        0.55 * image_array +
        0.45 * heatmap_colours
    ).astype(np.uint8)

    return image_array, heatmap_colours, overlay


# ==================================================
# STEP 6: CONVERT IMAGE TO BASE64
# ==================================================

def image_to_base64(image):

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded


# ==================================================
# STEP 7: GENERATE INSPECTION REPORT
# ==================================================

def generate_inspection_report(result):

    image = result["image"]

    cam = result["cam"]

    image_array, heatmap, overlay = create_overlay(
        image,
        cam
    )

    original_image = image_to_base64(
        Image.fromarray(image_array)
    )

    heatmap_image = image_to_base64(
        Image.fromarray(heatmap)
    )

    overlay_image = image_to_base64(
        Image.fromarray(overlay)
    )

    top_predictions_html = ""

    for name, score in result["top_predictions"]:

        top_predictions_html += f"""
        <tr>
            <td>{name}</td>
            <td>{score:.2f}%</td>
        </tr>
        """

    report = f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>Building Defect AI Inspection Report</title>

        <style>

            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                color: #263746;
            }}

            h1 {{
                color: #183B56;
            }}

            .card {{
                background: #F0F5F9;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}

            .images {{
                display: flex;
                gap: 20px;
            }}

            .images div {{
                width: 32%;
            }}

            .images img {{
                width: 100%;
                border-radius: 8px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
            }}

            td, th {{
                padding: 12px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}

            @media print {{

                button {{
                    display: none;
                }}

                body {{
                    margin: 15px;
                }}

            }}

        </style>

    </head>

    <body>

        <h1>Building Defect AI</h1>

        <h2>Building Surface Inspection Report</h2>

        <p>
            Inspection date:
            {datetime.now().strftime("%d %B %Y, %H:%M")}
        </p>

        <div class="card">

            <h2>Inspection Result</h2>

            <h3>
                Predicted Defect: {result["prediction"]}
            </h3>

            <h3>
                Confidence: {result["confidence"]:.2f}%
            </h3>

        </div>

        <h2>Uploaded Building Photograph</h2>

        <img
            src="data:image/png;base64,{original_image}"
            width="500"
        >

        <h2>Top Three Predictions</h2>

        <table>

            <tr>
                <th>Defect Category</th>
                <th>Confidence</th>
            </tr>

            {top_predictions_html}

        </table>

        <h2>AI Visual Explanation — Grad-CAM</h2>

        <div class="images">

            <div>

                <h3>Original Image</h3>

                <img src="data:image/png;base64,{original_image}">

            </div>

            <div>

                <h3>Attention Heatmap</h3>

                <img src="data:image/png;base64,{heatmap_image}">

            </div>

            <div>

                <h3>Attention Overlay</h3>

                <img src="data:image/png;base64,{overlay_image}">

            </div>

        </div>

        <p>
            Warm colours indicate regions with stronger influence
            on the model's prediction. They do not represent
            defect severity or precise defect boundaries.
        </p>

        <hr>

        <h3>Model Information</h3>

        <p>Model: ResNet18 — Experiment 3</p>

        <p>Number of Classes: 7</p>

        <p>Processing Device: {str(device).upper()}</p>

        <hr>

        <p>
            This report presents an AI image classification result.
            It is not a professional building-condition assessment.
        </p>

        <button onclick="window.print()">
            Print / Save as PDF
        </button>

    </body>

    </html>
    """

    return report


# ==================================================
# STEP 8: APPLICATION HEADER
# ==================================================

st.markdown(
    '<div class="main-title">Building Defect AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-assisted building surface inspection and defect classification'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

col1, col2 = st.columns(2)

with col1:

    st.caption(
        "MODEL: ResNet18 — Experiment 3"
    )

with col2:

    st.caption(
        f"PROCESSING: {str(device).upper()}"
    )

st.divider()


# ==================================================
# STEP 9: IMAGE UPLOAD
# ==================================================

st.subheader("Building Surface Inspection")

uploaded_file = st.file_uploader(
    "Upload a building surface photograph",
    type=["jpg", "jpeg", "png"],
    key="building_image"
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_column, result_column = st.columns(
        [1.2, 1],
        gap="large"
    )

    # ----------------------------------------------
    # LEFT: ORIGINAL IMAGE
    # ----------------------------------------------

    with image_column:

        st.markdown("### Uploaded Photograph")

        st.image(
            image,
            width=480
        )

        st.caption(
            f"Dimensions: {image.width} × {image.height} pixels"
        )

    # ----------------------------------------------
    # RIGHT: INSPECTION ANALYSIS
    # ----------------------------------------------

    with result_column:

        st.markdown("### Inspection Analysis")

        analyse_button = st.button(
            "Analyse Building Image",
            type="primary",
            use_container_width=True
        )

        if analyse_button:

            input_tensor = transform(
                image
            ).unsqueeze(0).to(device)

            # Model prediction

            with torch.no_grad():

                outputs = model(input_tensor)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                confidence, predicted_class = torch.max(
                    probabilities,
                    dim=1
                )

            predicted_index = predicted_class.item()

            predicted_name = class_names[
                predicted_index
            ].replace("_", " ").title()

            confidence_value = (
                confidence.item() * 100
            )

            # Top three predictions

            top_values, top_indices = torch.topk(
                probabilities,
                k=3
            )

            top_predictions = [

                (
                    class_names[index.item()].replace(
                        "_", " "
                    ).title(),

                    value.item() * 100

                )

                for value, index in zip(
                    top_values[0],
                    top_indices[0]
                )

            ]

            # Generate Grad-CAM

            cam = generate_gradcam(
                input_tensor,
                predicted_index
            )

            # Save inspection results

            st.session_state["inspection_result"] = {

                "cam": cam,

                "image": image,

                "prediction": predicted_name,

                "confidence": confidence_value,

                "top_predictions": top_predictions

            }

    # ==================================================
    # STEP 10: DISPLAY RESULTS
    # ==================================================

    if "inspection_result" in st.session_state:

        result = st.session_state["inspection_result"]

        # Avoid displaying old results for a new upload

        if result["image"] is not image and (
            result["image"].size != image.size
            or result["image"].tobytes() != image.tobytes()
        ):

            del st.session_state["inspection_result"]

        else:

            with result_column:

                st.divider()

                st.markdown("#### Predicted Defect")

                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="result-label">CLASSIFICATION</div>'
                    f'<div class="result-value">'
                    f'{result["prediction"]}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.metric(
                    "Prediction Confidence",
                    f'{result["confidence"]:.2f}%'
                )

                st.divider()

                st.markdown("#### Top 3 Predictions")

                for name, score in result["top_predictions"]:

                    st.write(f"**{name}**")

                    st.progress(
                        score / 100
                    )

                    st.caption(
                        f"{score:.2f}%"
                    )

            # ------------------------------------------
            # GRAD-CAM VISUALISATION
            # ------------------------------------------

            st.divider()

            st.subheader(
                "AI Visual Explanation — Grad-CAM"
            )

            image_array, heatmap, overlay = create_overlay(
                result["image"],
                result["cam"]
            )

            cam_col1, cam_col2, cam_col3 = st.columns(3)

            with cam_col1:

                st.markdown("**Original Image**")

                st.image(
                    image_array,
                    use_container_width=True
                )

            with cam_col2:

                st.markdown("**Attention Heatmap**")

                st.image(
                    heatmap,
                    use_container_width=True
                )

            with cam_col3:

                st.markdown("**Attention Overlay**")

                st.image(
                    overlay,
                    use_container_width=True
                )

            st.caption(
                "Warm colours indicate regions with stronger "
                "influence on the selected prediction. They do not "
                "represent defect severity or precise defect boundaries."
            )

            # ------------------------------------------
            # DOWNLOAD REPORT
            # ------------------------------------------

            st.divider()

            st.subheader("Inspection Report")

            report_html = generate_inspection_report(
                result
            )

            st.download_button(
                label="Download Inspection Report",
                data=report_html,
                file_name="building_defect_inspection_report.html",
                mime="text/html",
                use_container_width=True
            )


else:

    # Clear previous inspection when no image is uploaded

    st.session_state.pop(
        "inspection_result",
        None
    )

    st.info(
        "Upload a building photograph to begin inspection."
    )


# ==================================================
# STEP 11: FOOTER
# ==================================================

st.divider()

st.caption(
    "Building Defect AI | ResNet18 Transfer Learning | IAAC AI Methods Project"
)