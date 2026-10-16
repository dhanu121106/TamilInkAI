
import os
from io import BytesIO
from main import extract_tamil
from dotenv import load_dotenv
import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image, ImageOps
load_dotenv(override=True)
# ============================================================
# TAMILINK AI - CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TamilInk AI",
    page_icon="📜",
    layout="wide",
)

WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(WINDOWS_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_variants(image):
    """Prepare multiple image variants for Tamil OCR."""

    rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Upscale small documents
    if max(gray.shape) < 2000:
        gray = cv2.resize(
            gray, None, fx=2, fy=2,
            interpolation=cv2.INTER_CUBIC
        )

    denoised = cv2.fastNlMeansDenoising(gray, None, h=8)

    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(denoised)

    _, otsu = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 11
    )

    # Optional notebook-line removal
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(30, gray.shape[1] // 20), 1)
    )

    horizontal_lines = cv2.morphologyEx(
        otsu, cv2.MORPH_OPEN, kernel
    )

    line_removed = cv2.subtract(otsu, horizontal_lines)

    return {
        "Upscaled grayscale": gray,
        "Denoised grayscale": denoised,
        "Enhanced grayscale": enhanced,
        "Otsu threshold": otsu,
        "Adaptive threshold": adaptive,
        "Notebook lines removed": line_removed,
    }


def restore_document(image, denoise_strength=10, contrast=2.0, threshold=7):
    """Restore document readability."""

    rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if max(gray.shape) < 2000:
        gray = cv2.resize(
            gray, None, fx=2, fy=2,
            interpolation=cv2.INTER_CUBIC
        )

    denoised = cv2.fastNlMeansDenoising(
        gray, None,
        h=int(denoise_strength),
        templateWindowSize=7,
        searchWindowSize=21
    )

    enhanced = cv2.createCLAHE(
        clipLimit=float(contrast),
        tileGridSize=(8, 8)
    ).apply(denoised)

    restored = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, int(threshold)
    )

    return Image.fromarray(restored)


# ============================================================
# TAMIL OCR - MULTIPLE VARIANTS AND LAYOUTS
# ============================================================

def recognize_best(image, psm=None):
    """
    Run OCR on multiple image variants and page layouts.
    Rank results by mean Tesseract word confidence.
    Confidence is not verified transcription accuracy.
    """

    results = []
    variants = prepare_variants(image)
    page_modes = [6, 11, 4] if psm is None else [int(psm)]

    for variant_name, array in variants.items():
        for page_mode in page_modes:
            config = f"--oem 1 --psm {page_mode}"

            try:
                data = pytesseract.image_to_data(
                    array,
                    lang="tam",
                    config=config,
                    output_type=pytesseract.Output.DICT
                )

                scores = []

                for confidence, word in zip(
                    data["conf"], data["text"]
                ):
                    if word.strip():
                        try:
                            score = float(confidence)
                            if score >= 0:
                                scores.append(score)
                        except (ValueError, TypeError):
                            pass

                text = pytesseract.image_to_string(
                    array,
                    lang="tam",
                    config=config
                ).strip()

                if text:
                    average = (
                        sum(scores) / len(scores)
                        if scores else 0.0
                    )

                    results.append(
                        (average, variant_name, page_mode, text)
                    )

            except (pytesseract.TesseractError, RuntimeError):
                continue

    results.sort(key=lambda item: item[0], reverse=True)
    return results


# ============================================================
# CHECK TESSERACT
# ============================================================

def check_tesseract():
    try:
        version = str(pytesseract.get_tesseract_version())
        languages = pytesseract.get_languages(config="")
        return version, languages, None
    except Exception as exc:
        return None, [], str(exc)


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("📜 TamilInk AI")

st.subheader(
    "Automated Restoration and Recognition of Handwritten Tamil Documents"
)

st.write(
    "Restore faded or noisy Tamil documents, compare OCR results, "
    "edit extracted text, and download the transcription."
)


# -------------------- SIDEBAR SETTINGS --------------------

st.sidebar.header("Restoration Settings")

denoise_strength = st.sidebar.slider(
    "Noise removal", 1, 20, 10
)

contrast = st.sidebar.slider(
    "Contrast enhancement", 1.0, 5.0, 2.0, 0.5
)

threshold = st.sidebar.slider(
    "Threshold adjustment", 0, 20, 7
)

st.sidebar.divider()
st.sidebar.header("OCR Settings")
ocr_engine = st.sidebar.selectbox(
    "OCR Engine",
    [
        "TamilInk AI",
        "Tesseract (Offline)"
    ]
)

layout_choice = st.sidebar.selectbox(
    "Page layout",
    [
        "Try multiple layouts",
        "Uniform text block (6)",
        "Scattered text (11)",
        "Column text (4)",
    ]
)

psm_map = {
    "Try multiple layouts": None,
    "Uniform text block (6)": 6,
    "Scattered text (11)": 11,
    "Column text (4)": 4,
}

selected_psm = psm_map[layout_choice]

st.sidebar.caption("Free local processing. No API key required.")


# -------------------- OCR STATUS --------------------

version, languages, error = check_tesseract()

if error:
    st.error("Tesseract could not be started. Check its installation path.")
    st.code(error)

elif "tam" not in languages:
    st.warning(
        "Tamil language data ('tam') was not found. "
        "Install the Tamil language data before running OCR."
    )

else:
    st.success("Offline Tamil OCR is ready.")

    with st.expander("OCR engine details"):
        st.write(f"Tesseract version: {version}")
        st.write("Installed languages: " + ", ".join(languages))


# -------------------- DOCUMENT UPLOAD --------------------

uploaded_file = st.file_uploader(
    "Upload a handwritten Tamil document",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:

    try:
        original = Image.open(uploaded_file)
        original = ImageOps.exif_transpose(original).convert("RGB")

        st.caption(
            f"Image dimensions: {original.width} × {original.height} pixels"
        )

        restored = restore_document(
            original,
            denoise_strength,
            contrast,
            threshold
        )

        left, right = st.columns(2)

        with left:
            st.markdown("### Original Document")
            st.image(original, width="stretch")

        with right:
            st.markdown("### Restored Document")
            st.image(restored, width="stretch")

            image_buffer = BytesIO()
            restored.save(image_buffer, format="PNG")

            st.download_button(
                "Download Restored Image",
                data=image_buffer.getvalue(),
                file_name="tamilink_restored.png",
                mime="image/png"
            )

        st.divider()
        st.markdown("### Tamil Text Recognition")

        st.write(
            "OCR runs on the original image using multiple preprocessing "
            "variants. The restored image is available for visual review."
        )

        
        # -------------------- RUN OCR --------------------

        if st.button("Recognize Tamil Text", type="primary"):

            if ocr_engine == "TamilInk AI":

                if not os.getenv("GEMINI_API_KEY"):
                    st.error("GEMINI_API_KEY is missing from .env.")

                else:
                    try:
                        with st.spinner(
                            "TamilInk AI is transcribing the Tamil document..."
                        ):
                            image_buffer = BytesIO()
                            original.save(
                                image_buffer,
                                format="JPEG",
                                quality=95
                            )

                            recognized = extract_tamil(
                                image_buffer.getvalue()
                            )

                        st.session_state["recognized_text"] = recognized
                        st.session_state["ocr_image_name"] = uploaded_file.name
                        st.session_state["ocr_candidates"] = []

                        st.success("TamilInk AI transcription completed.")

                    except Exception as exc:
                        st.error(f"TamilInk AI OCR failed: {exc}")

            else:
                if error:
                    st.error("Fix the Tesseract installation first.")

                elif "tam" not in languages:
                    st.error(
                        "Tesseract Tamil language data ('tam') is missing."
                    )

                else:
                    try:
                        with st.spinner(
                            "Testing preprocessing variants and OCR layouts..."
                        ):
                            results = recognize_best(
                                original,
                                psm=selected_psm
                            )

                        st.session_state["ocr_candidates"] = results
                        st.session_state["ocr_image_name"] = uploaded_file.name
                        st.session_state["recognized_text"] = (
                            results[0][3] if results else ""
                        )

                        if results:
                            confidence, name, page_mode, _ = results[0]

                            st.success(f"Best preprocessing: {name}")
                            st.caption(
                                f"Selected page layout: PSM {page_mode}"
                            )
                            st.caption(
                                f"Mean Tesseract confidence: "
                                f"{confidence:.1f}/100. "
                                "This is not a measured accuracy percentage."
                            )
                        else:
                            st.warning(
                                "No text was recognized. Try a clearer image "
                                "or a different page layout."
                            )

                    except Exception as exc:
                        st.error(f"OCR failed: {exc}")

        

        # -------------------- EDIT AND DOWNLOAD TEXT --------------------

        if (
            "recognized_text" in st.session_state
            and st.session_state.get("ocr_image_name") == uploaded_file.name
        ):

            st.markdown("### Review and Correct Tamil Text")

            st.caption(
                "Verify the extracted text against the original document."
            )

            corrected_text = st.text_area(
                "Editable Tamil transcription",
                value=st.session_state["recognized_text"],
                height=250,
                key=f"corrected_{uploaded_file.name}"
            )

            st.download_button(
                "Download Tamil Text",
                data=corrected_text.encode("utf-8"),
                file_name="tamilink_transcription.txt",
                mime="text/plain"
            )

            # -------------------- ALTERNATIVE OCR RESULTS --------------------

            candidates = st.session_state.get("ocr_candidates", [])

            if len(candidates) > 1:
                st.markdown("### Compare Alternative OCR Results")

                for confidence, name, page_mode, text in candidates[1:]:
                    with st.expander(
                        f"{name} — PSM {page_mode} — "
                        f"confidence {confidence:.1f}/100"
                    ):
                        st.text(text)

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

else:
    st.info("Upload a document image to begin restoration and recognition.")


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "TamilInk AI | Free offline prototype | "
    "OCR results require human verification."
)
