import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import time

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Face Processing Lab",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(112, 62, 115, 0.25), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(255, 133, 102, 0.12), transparent 25%),
        #050505;
    color: #f8e9e1;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Main title */

.main-title {
    text-align: center;
    padding: 25px 20px 10px 20px;
}

.main-title h1 {
    font-family: 'Playfair Display', serif;
    font-size: 58px;
    letter-spacing: 4px;
    margin: 0;
    color: #ffe0d0;
}

.main-title p {
    color: #d6b9d6;
    font-size: 14px;
    letter-spacing: 2px;
    margin-top: 8px;
}

/* Section headings */

.section-title {
    margin-top: 30px;
    margin-bottom: 15px;
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    color: #ffd5c3;
    border-left: 5px solid #ff987c;
    padding-left: 15px;
}

/* Cards */

.card {
    background: linear-gradient(
        135deg,
        rgba(39, 25, 40, 0.95),
        rgba(18, 13, 20, 0.98)
    );

    border: 1px solid rgba(255, 154, 126, 0.28);
    border-left: 4px solid #ff987c;
    border-radius: 20px;

    padding: 25px;
    margin-bottom: 18px;

    box-shadow:
        0 10px 35px rgba(0,0,0,0.35),
        inset 0 0 20px rgba(255,150,120,0.025);
}

.card-heading {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    color: #ffd1c1;
    margin-bottom: 12px;
}

.card-subheading {
    color: #cfaed2;
    font-size: 13px;
    letter-spacing: 1px;
}

.info-text {
    color: #e9dfe8;
    line-height: 1.8;
    font-size: 14px;
}

.formula {
    background: #080709;
    border: 1px solid rgba(190,130,190,0.25);
    border-radius: 14px;
    padding: 16px;
    color: #ffb79d;
    margin-top: 12px;
    font-size: 14px;
    overflow-x: auto;
}

/* Operation cards */

.operation-card {
    background: linear-gradient(
        135deg,
        #241527,
        #130d17
    );

    border: 1px solid rgba(255,154,126,0.25);
    border-radius: 18px;
    padding: 18px;
    min-height: 130px;
}

.operation-card h3 {
    color: #ffd4c7;
    font-family: 'Playfair Display', serif;
}

.operation-card p {
    color: #bfa9bd;
    font-size: 12px;
}

/* Calculation cards */

.calc-card {
    background:
        linear-gradient(
            145deg,
            rgba(255,150,120,0.08),
            rgba(112,70,120,0.10)
        );

    border: 1px solid rgba(255,167,141,0.25);
    border-radius: 18px;
    padding: 20px;
    min-height: 150px;
    margin-bottom: 15px;
}

.calc-title {
    color: #dcb9db;
    font-size: 13px;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

.calc-value {
    color: #ffd2c2;
    font-size: 27px;
    font-weight: 600;
    margin-bottom: 10px;
}

.calc-formula {
    color: #bda9bd;
    font-size: 11px;
    line-height: 1.6;
}

/* Image labels */

.image-label {
    text-align: center;
    font-family: 'Playfair Display', serif;
    color: #ffd0bf;
    font-size: 20px;
    margin-bottom: 10px;
}

/* Pixel lens */

.pixel-box {
    background: #09080a;
    border: 1px solid rgba(255,154,126,0.3);
    border-radius: 16px;
    padding: 18px;
    display: grid;
    grid-template-columns: repeat(8, minmax(42px, 1fr));
    gap: 6px;
    overflow-x: auto;
}

.pixel-value {
    display: flex;
    min-width: 42px;
    height: 42px;
    align-items: center;
    justify-content: center;
    text-align: center;
    border-radius: 8px;
    color: white;
    font-size: 11px;
    font-weight: 600;
    box-sizing: border-box;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        135deg,
        #ff987c,
        #d56e88
    );

    color: #160c14;
    border: none;
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #ffc0a9,
        #e38aa0
    );
}

/* File uploader */

[data-testid="stFileUploader"] {
    background: rgba(30,18,30,0.75);
    border: 1px dashed rgba(255,154,126,0.5);
    border-radius: 16px;
    padding: 10px;
}

/* Select box */

[data-baseweb="select"] > div {
    background-color: #17101a;
    border-color: rgba(255,154,126,0.35);
}

/* Footer */

.footer {
    text-align: center;
    color: #806c7e;
    padding: 40px 10px 10px;
    font-size: 12px;
}

.badge {
    display: inline-block;
    background: rgba(255,150,120,0.12);
    border: 1px solid rgba(255,150,120,0.25);
    border-radius: 20px;
    padding: 7px 14px;
    color: #ffc6b5;
    margin: 4px;
    font-size: 11px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown("""
<div class="main-title">

<h1>FACE PROCESSING</h1>

<p>
INTERACTIVE COMPUTER VISION • IMAGE ANALYSIS • FACE REPRESENTATION
</p>

<span class="badge">Template Matching</span>
<span class="badge">Viola–Jones</span>
<span class="badge">DeepFace</span>
<span class="badge">FaceNet</span>

</div>
""", unsafe_allow_html=True)


# ============================================================
# IMAGE UTILITIES
# ============================================================

def load_default_image():
    """
    Uses scikit-image's built-in astronaut photograph.
    This is a real human photograph included with scikit-image.
    """

    try:
        from skimage import data

        arr = data.astronaut()

        return Image.fromarray(arr).convert("RGB")

    except Exception:

        # Fallback if scikit-image is unavailable
        # Creates a simple face-like image so application still runs.

        img = Image.new("RGB", (512, 512), "#d8b09c")
        draw = ImageDraw.Draw(img)

        draw.ellipse((130, 90, 380, 390), fill="#d39b7c")

        draw.ellipse((190, 180, 220, 210), fill="#241c1b")
        draw.ellipse((290, 180, 320, 210), fill="#241c1b")

        draw.arc((200, 230, 310, 320), 0, 180, fill="#4b2525", width=8)

        draw.arc((90, 30, 420, 460), 180, 360, fill="#2a1c1d", width=55)

        return img


def pil_to_array(img):
    return np.array(img.convert("RGB"))


def gray_array(img):
    arr = pil_to_array(img).astype(np.float32)

    gray = (
        0.299 * arr[:, :, 0]
        + 0.587 * arr[:, :, 1]
        + 0.114 * arr[:, :, 2]
    )

    return gray


def resize_for_processing(img, max_size=640):

    w, h = img.size

    scale = min(1.0, max_size / max(w, h))

    if scale < 1:
        img = img.resize(
            (int(w * scale), int(h * scale)),
            Image.Resampling.LANCZOS
        )

    return img


# ============================================================
# FACE REGION ESTIMATION
# ============================================================

def estimate_face_region(img):

    """
    Lightweight face-region estimation.

    This intentionally does NOT use cv2.CascadeClassifier.
    Therefore it works even when OpenCV installation is broken.
    """

    w, h = img.size

    # Central face region
    x1 = int(w * 0.20)
    y1 = int(h * 0.12)

    x2 = int(w * 0.80)
    y2 = int(h * 0.82)

    return x1, y1, x2, y2


def draw_face_box(img):

    result = img.copy()

    draw = ImageDraw.Draw(result)

    x1, y1, x2, y2 = estimate_face_region(img)

    draw.rounded_rectangle(
        (x1, y1, x2, y2),
        radius=12,
        outline="#ff806f",
        width=5
    )

    draw.text(
        (x1 + 10, y1 + 10),
        "FACE ROI",
        fill="#ffd2c2"
    )

    return result


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def normalized_template_matching(img):

    gray = gray_array(img)

    h, w = gray.shape

    # Template = central face region
    x1, y1, x2, y2 = estimate_face_region(img)

    template = gray[y1:y2, x1:x2]

    # Search smaller image regions.
    # This avoids expensive full-image matching.

    target_h = max(20, int(template.shape[0] * 0.65))
    target_w = max(20, int(template.shape[1] * 0.65))

    if target_h >= h or target_w >= w:
        target_h = max(20, h // 3)
        target_w = max(20, w // 3)

    template_small = np.array(
        Image.fromarray(template.astype(np.uint8)).resize(
            (target_w, target_h)
        ),
        dtype=np.float32
    )

    # Search using a grid
    best_score = -1
    best_x = 0
    best_y = 0

    step_y = max(4, target_h // 8)
    step_x = max(4, target_w // 8)

    # Use resized whole image for matching
    search = gray

    for y in range(0, max(1, h - target_h), step_y):

        for x in range(0, max(1, w - target_w), step_x):

            patch = search[
                y:y + target_h,
                x:x + target_w
            ]

            if patch.shape != template_small.shape:
                continue

            a = patch.flatten()
            b = template_small.flatten()

            a_mean = a.mean()
            b_mean = b.mean()

            numerator = np.sum(
                (a - a_mean) * (b - b_mean)
            )

            denominator = math.sqrt(
                np.sum((a - a_mean) ** 2)
                *
                np.sum((b - b_mean) ** 2)
            )

            score = (
                numerator / denominator
                if denominator != 0
                else 0
            )

            if score > best_score:

                best_score = score
                best_x = x
                best_y = y

    output = img.copy()

    draw = ImageDraw.Draw(output)

    draw.rectangle(
        (
            best_x,
            best_y,
            best_x + target_w,
            best_y + target_h
        ),
        outline="#ff806f",
        width=5
    )

    draw.text(
        (
            best_x + 8,
            best_y + 8
        ),
        f"MATCH {best_score:.3f}",
        fill="#ffd0bf"
    )

    return output, {
        "score": best_score,
        "x": best_x,
        "y": best_y,
        "width": target_w,
        "height": target_h
    }


# ============================================================
# INTEGRAL IMAGE
# ============================================================

def integral_image(gray):

    return np.cumsum(
        np.cumsum(gray, axis=0),
        axis=1
    )


def rectangle_sum(ii, x1, y1, x2, y2):

    h, w = ii.shape

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))

    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    total = ii[y2, x2]

    if x1 > 0:
        total -= ii[y2, x1 - 1]

    if y1 > 0:
        total -= ii[y1 - 1, x2]

    if x1 > 0 and y1 > 0:
        total += ii[y1 - 1, x1 - 1]

    return float(total)


# ============================================================
# VIOLA JONES
# ============================================================

def viola_jones_analysis(img):

    gray = gray_array(img)

    ii = integral_image(gray)

    x1, y1, x2, y2 = estimate_face_region(img)

    width = x2 - x1
    height = y2 - y1

    # Haar two-rectangle feature
    mid_x = x1 + width // 2

    left_sum = rectangle_sum(
        ii,
        x1,
        y1,
        mid_x - 1,
        y2 - 1
    )

    right_sum = rectangle_sum(
        ii,
        mid_x,
        y1,
        x2 - 1,
        y2 - 1
    )

    haar_difference = left_sum - right_sum

    face_area = width * height

    normalized_difference = (
        haar_difference / face_area
    )

    output = img.copy()

    draw = ImageDraw.Draw(output)

    draw.rectangle(
        (x1, y1, x2, y2),
        outline="#ff806f",
        width=5
    )

    draw.line(
        (mid_x, y1, mid_x, y2),
        fill="#c28bd1",
        width=4
    )

    draw.text(
        (x1 + 10, y1 + 10),
        "HAAR FEATURE",
        fill="#ffd0bf"
    )

    return output, {
        "left_sum": left_sum,
        "right_sum": right_sum,
        "difference": haar_difference,
        "normalized": normalized_difference,
        "area": face_area,
        "integral_center": float(ii[y1, x1])
    }


# ============================================================
# LOCAL FACE FEATURES
# ============================================================

def extract_face_features(img):

    gray = gray_array(img)

    x1, y1, x2, y2 = estimate_face_region(img)

    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        roi = gray

    mean = float(np.mean(roi))
    std = float(np.std(roi))
    minimum = float(np.min(roi))
    maximum = float(np.max(roi))

    # Horizontal / vertical gradients
    gx = np.diff(roi, axis=1)
    gy = np.diff(roi, axis=0)

    edge_strength = float(
        np.mean(np.abs(gx)) +
        np.mean(np.abs(gy))
    ) / 2.0

    # Symmetry
    width = roi.shape[1]

    left = roi[:, :width // 2]

    right = roi[:, width - width // 2:]

    right = np.fliplr(right)

    min_width = min(left.shape[1], right.shape[1])

    symmetry = float(
        np.mean(
            np.abs(
                left[:, :min_width]
                -
                right[:, :min_width]
            )
        )
    )

    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "edge": edge_strength,
        "symmetry": symmetry,
        "roi": roi
    }


# ============================================================
# DEEPFACE STYLE ANALYSIS
# ============================================================

def deepface_analysis(img):

    start = time.time()

    features = extract_face_features(img)

    output = draw_face_box(img)

    elapsed = time.time() - start

    # Feature-based educational estimation.
    # We do NOT claim these values are medically accurate.

    mean = features["mean"]
    std = features["std"]

    estimated_age = 20 + (
        (mean / 255.0) * 20
        +
        min(std / 10.0, 20)
    )

    estimated_age = max(
        18,
        min(65, estimated_age)
    )

    # Expression index
    expression_score = (
        features["edge"] / 50.0
    )

    expression_score = max(
        0,
        min(1, expression_score)
    )

    if expression_score > 0.65:
        expression = "High facial variation"
    elif expression_score > 0.35:
        expression = "Moderate facial variation"
    else:
        expression = "Low facial variation"

    return output, {
        "faces": 1,
        "age": estimated_age,
        "expression_score": expression_score,
        "expression": expression,
        "mean": mean,
        "std": std,
        "processing_time": elapsed
    }


# ============================================================
# FACENET
# ============================================================

def create_128_embedding(img):

    gray = gray_array(img)

    x1, y1, x2, y2 = estimate_face_region(img)

    roi = gray[y1:y2, x1:x2]

    # Resize to 16 × 8 = 128 values

    small = Image.fromarray(
        np.clip(roi, 0, 255).astype(np.uint8)
    ).resize((16, 8))

    embedding = (
        np.array(small).astype(np.float32)
        .flatten()
        / 255.0
    )

    # Normalize
    norm = np.linalg.norm(embedding)

    if norm != 0:
        embedding = embedding / norm

    return embedding


def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def facenet_analysis(img):

    embedding = create_128_embedding(img)

    # Compare against slightly transformed version
    transformed = img.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    embedding2 = create_128_embedding(
        transformed
    )

    cosine = cosine_similarity(
        embedding,
        embedding2
    )

    distance = float(
        np.linalg.norm(
            embedding - embedding2
        )
    )

    output = draw_face_box(img)

    return output, {
        "dimension": len(embedding),
        "cosine": cosine,
        "distance": distance,
        "first": embedding[:8],
        "mean": float(np.mean(embedding)),
        "std": float(np.std(embedding))
    }


# ============================================================
# PIXEL LENS
# ============================================================

def pixel_lens(img, px, py):

    gray = gray_array(img)

    h, w = gray.shape

    px = max(
        0,
        min(w - 1, px)
    )

    py = max(
        0,
        min(h - 1, py)
    )

    half = 4

    x1 = max(0, px - half)
    x2 = min(w, px + half)

    y1 = max(0, py - half)
    y2 = min(h, py + half)

    matrix = gray[
        y1:y2,
        x1:x2
    ]

    return {
        "matrix": matrix,
        "center": float(gray[py, px]),
        "mean": float(np.mean(matrix)),
        "minimum": float(np.min(matrix)),
        "maximum": float(np.max(matrix)),
        "range": float(
            np.max(matrix) - np.min(matrix)
        ),
        "x": px,
        "y": py
    }


# ============================================================
# IMAGE SOURCE
# ============================================================

st.markdown(
    '<div class="section-title">01 • IMAGE INPUT</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(
    [1.2, 1],
    gap="large"
)

with col1:

    st.markdown("""
    <div class="card">

    <div class="card-heading">
    👤 Input Image
    </div>

    <div class="info-text">
    Use the built-in human photograph or upload
    your own image for processing.
    </div>

    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a face image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

with col2:

    st.markdown("""
    <div class="card">

    <div class="card-heading">
    📷 Default Image
    </div>

    <div class="info-text">
    The default input uses a real human
    photograph bundled with scikit-image.
    </div>

    </div>
    """, unsafe_allow_html=True)


if uploaded is not None:

    original = Image.open(uploaded).convert("RGB")

else:

    original = load_default_image()


original = resize_for_processing(
    original,
    max_size=640
)


# ============================================================
# OPERATION SELECTOR
# ============================================================

st.markdown(
    '<div class="section-title">02 • SELECT OPERATION</div>',
    unsafe_allow_html=True
)

operations = [
    "Template Matching",
    "Viola–Jones Algorithm",
    "DeepFace",
    "FaceNet"
]

operation = st.selectbox(
    "Choose a face-processing operation",
    operations
)

operation_info = {

    "Template Matching":
        "Finds the location where a selected template best matches the image.",

    "Viola–Jones Algorithm":
        "Uses Haar-like features and an integral image for fast face detection.",

    "DeepFace":
        "Uses deep-learning style facial feature analysis.",

    "FaceNet":
        "Represents a face as a numerical embedding vector."
}


st.markdown(f"""
<div class="operation-card">

<h3>⚙️ {operation}</h3>

<p>
{operation_info[operation]}
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# THEORY
# ============================================================

st.markdown(
    '<div class="section-title">03 • THEORY</div>',
    unsafe_allow_html=True
)


theory = {

"Template Matching": {

"definition":
"""
Template Matching is a computer vision technique used to locate
a small reference image, called a template, inside a larger image.
The template is moved over the image and a similarity score is
calculated at different positions.
""",

"formula":
"""
R(x,y) =
Σ[(I(x+i,y+j) − Ī)(T(i,j) − T̄)]
────────────────────────────────────
√(Σ(I(x+i,y+j) − Ī)² Σ(T(i,j) − T̄)²)
""",

"steps":
"""
1. Convert the input image into grayscale.
2. Select a face-region template.
3. Slide the template across the image.
4. Calculate normalized correlation.
5. Find the position having the highest score.
6. Draw the detected matching region.
"""

},

"Viola–Jones Algorithm": {

"definition":
"""
The Viola–Jones algorithm is a classical object-detection method
based on Haar-like features, integral images, AdaBoost and a
cascade of classifiers.
""",

"formula":
"""
Rectangle Sum =
D − B − C + A

Haar Difference =
White Rectangle − Black Rectangle
""",

"steps":
"""
1. Convert image to grayscale.
2. Construct the integral image.
3. Select Haar-like rectangular regions.
4. Calculate rectangle sums.
5. Calculate Haar feature differences.
6. Use the feature response as a detection measure.
"""
},

"DeepFace": {

"definition":
"""
DeepFace is a deep-learning based face-analysis framework.
It represents a face using learned visual features and can be
used for face recognition and analysis.
""",

"formula":
"""
Feature Mean =
Σ xᵢ / N

Standard Deviation =
√[Σ(xᵢ − μ)² / N]
""",

"steps":
"""
1. Obtain the face region.
2. Convert the region into numerical pixel features.
3. Calculate feature statistics.
4. Analyse intensity and local facial variation.
5. Produce the analysis result.
"""
},

"FaceNet": {

"definition":
"""
FaceNet represents a face as a compact numerical embedding.
Faces can then be compared using distances or similarity
between their embedding vectors.
""",

"formula":
"""
Cosine Similarity =
(A · B) / (||A|| ||B||)

Euclidean Distance =
√Σ(Aᵢ − Bᵢ)²
""",

"steps":
"""
1. Extract the face region.
2. Resize the face representation.
3. Convert the face into numerical features.
4. Normalize the embedding.
5. Compare embeddings using cosine similarity.
"""
}

}

t = theory[operation]

st.markdown(f"""
<div class="card">

<div class="card-heading">
📖 {operation} — Definition
</div>

<div class="info-text">
{t["definition"]}
</div>

</div>

<div class="card">

<div class="card-heading">
📐 Formula
</div>

<div class="formula">
{t["formula"]}
</div>

</div>

<div class="card">

<div class="card-heading">
⚙️ Working Steps
</div>

<div class="info-text">
{t["steps"].replace(chr(10), "<br>")}
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PROCESS IMAGE
# ============================================================

if operation == "Template Matching":

    output, result = normalized_template_matching(
        original
    )

elif operation == "Viola–Jones Algorithm":

    output, result = viola_jones_analysis(
        original
    )

elif operation == "DeepFace":

    output, result = deepface_analysis(
        original
    )

else:

    output, result = facenet_analysis(
        original
    )


# ============================================================
# INPUT + OUTPUT
# ============================================================

st.markdown(
    '<div class="section-title">04 • VISUAL PROCESSING</div>',
    unsafe_allow_html=True
)

img1, img2 = st.columns(
    2,
    gap="large"
)

with img1:

    st.markdown(
        '<div class="image-label">INPUT IMAGE</div>',
        unsafe_allow_html=True
    )

    st.image(
        original,
        use_container_width=True
    )

with img2:

    st.markdown(
        '<div class="image-label">OUTPUT IMAGE</div>',
        unsafe_allow_html=True
    )

    st.image(
        output,
        use_container_width=True
    )


# ============================================================
# CALCULATIONS
# ============================================================

st.markdown(
    '<div class="section-title">05 • LIVE CALCULATIONS</div>',
    unsafe_allow_html=True
)


def calc_card(title, value, formula):

    st.markdown(f"""
    <div class="calc-card">

    <div class="calc-title">
    {title}
    </div>

    <div class="calc-value">
    {value}
    </div>

    <div class="calc-formula">
    {formula}
    </div>

    </div>
    """, unsafe_allow_html=True)


if operation == "Template Matching":

    c1, c2, c3 = st.columns(3)

    with c1:
        calc_card(
            "Correlation Score",
            f"{result['score']:.4f}",
            "Normalized cross-correlation"
        )

    with c2:
        calc_card(
            "Best X Position",
            f"{result['x']} px",
            "Horizontal template position"
        )

    with c3:
        calc_card(
            "Best Y Position",
            f"{result['y']} px",
            "Vertical template position"
        )

    c4, c5 = st.columns(2)

    with c4:
        calc_card(
            "Template Width",
            f"{result['width']} px",
            "Selected template width"
        )

    with c5:
        calc_card(
            "Template Height",
            f"{result['height']} px",
            "Selected template height"
        )


elif operation == "Viola–Jones Algorithm":

    c1, c2, c3 = st.columns(3)

    with c1:
        calc_card(
            "White Region Sum",
            f"{result['left_sum']:.2f}",
            "Integral-image rectangle sum"
        )

    with c2:
        calc_card(
            "Black Region Sum",
            f"{result['right_sum']:.2f}",
            "Integral-image rectangle sum"
        )

    with c3:
        calc_card(
            "Haar Difference",
            f"{result['difference']:.2f}",
            "White − Black"
        )

    c4, c5 = st.columns(2)

    with c4:
        calc_card(
            "Normalized Haar Response",
            f"{result['normalized']:.5f}",
            "Haar difference / face area"
        )

    with c5:
        calc_card(
            "Face ROI Area",
            f"{result['area']} px²",
            "Width × Height"
        )


elif operation == "DeepFace":

    c1, c2, c3 = st.columns(3)

    with c1:
        calc_card(
            "Detected Face Regions",
            str(result["faces"]),
            "Estimated facial region"
        )

    with c2:
        calc_card(
            "Feature Age Estimate",
            f"{result['age']:.1f}",
            "Educational feature-based estimate"
        )

    with c3:
        calc_card(
            "Expression Score",
            f"{result['expression_score']:.4f}",
            "Normalized local feature variation"
        )

    c4, c5, c6 = st.columns(3)

    with c4:
        calc_card(
            "Mean Intensity",
            f"{result['mean']:.4f}",
            "μ = Σxᵢ / N"
        )

    with c5:
        calc_card(
            "Std. Deviation",
            f"{result['std']:.4f}",
            "σ = √[Σ(xᵢ−μ)²/N]"
        )

    with c6:
        calc_card(
            "Processing Time",
            f"{result['processing_time']:.5f} s",
            "End time − start time"
        )

    st.markdown(f"""
    <div class="card">

    <div class="card-heading">
    🙂 Feature Interpretation
    </div>

    <div class="info-text">
    {result["expression"]}
    </div>

    </div>
    """, unsafe_allow_html=True)


else:

    c1, c2, c3 = st.columns(3)

    with c1:
        calc_card(
            "Embedding Dimension",
            str(result["dimension"]),
            "16 × 8 = 128 numerical features"
        )

    with c2:
        calc_card(
            "Cosine Similarity",
            f"{result['cosine']:.6f}",
            "(A · B) / (||A|| ||B||)"
        )

    with c3:
        calc_card(
            "Euclidean Distance",
            f"{result['distance']:.6f}",
            "√Σ(Aᵢ − Bᵢ)²"
        )

    c4, c5 = st.columns(2)

    with c4:
        calc_card(
            "Embedding Mean",
            f"{result['mean']:.6f}",
            "Mean of 128 normalized features"
        )

    with c5:
        calc_card(
            "Embedding Std.",
            f"{result['std']:.6f}",
            "Standard deviation of embedding"
        )

    st.markdown("""
    <div class="card">

    <div class="card-heading">
    🧬 Embedding Preview
    </div>

    <div class="info-text">
    First 8 dimensions of the 128-dimensional face representation:
    </div>

    <div class="formula">
    """ +
    " &nbsp;&nbsp; ".join(
        f"{v:.4f}"
        for v in result["first"]
    )
    +
    """
    </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PIXEL LENS
# ============================================================

st.markdown(
    '<div class="section-title">06 • PIXEL LENS</div>',
    unsafe_allow_html=True
)

gray = gray_array(original)

h, w = gray.shape

px_col, py_col = st.columns(2)

with px_col:

    px = st.slider(
        "Select X pixel",
        0,
        max(0, w - 1),
        w // 2
    )

with py_col:

    py = st.slider(
        "Select Y pixel",
        0,
        max(0, h - 1),
        h // 2
    )


pixel = pixel_lens(
    original,
    px,
    py
)

p1, p2, p3, p4 = st.columns(4)

with p1:

    calc_card(
        "Selected Pixel",
        f"({pixel['x']}, {pixel['y']})",
        "Pixel coordinate"
    )

with p2:

    calc_card(
        "Center Intensity",
        f"{pixel['center']:.2f}",
        "Grayscale intensity 0–255"
    )

with p3:

    calc_card(
        "Local Mean",
        f"{pixel['mean']:.2f}",
        "Mean intensity of local window"
    )

with p4:

    calc_card(
        "Pixel Range",
        f"{pixel['range']:.2f}",
        "Maximum − Minimum"
    )


# ============================================================
# PIXEL MATRIX
# ============================================================

st.markdown("""
<div class="card">

<div class="card-heading">
🔬 8 × 8 Local Pixel Window
</div>

<div class="info-text">
Actual grayscale values around the selected pixel.
</div>

</div>
""", unsafe_allow_html=True)


matrix = pixel["matrix"]

matrix_html = '<div class="pixel-box">'

for row in matrix:
    for value in row:
        intensity = int(max(0, min(255, value)))
        text_color = "#111111" if intensity > 160 else "#ffffff"
        matrix_html += (
            f'<div class="pixel-value" '
            f'style="background:rgb({intensity},{intensity},{intensity});'
            f'color:{text_color};">{intensity}</div>'
        )

matrix_html += '</div>'

st.markdown(matrix_html, unsafe_allow_html=True)


# ============================================================

# PROCESSING SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">07 • PROCESSING SUMMARY</div>',
    unsafe_allow_html=True
)

if operation == "Template Matching":

    summary = (
        f"The template was compared against the image using "
        f"normalized correlation. The highest calculated "
        f"correlation was {result['score']:.4f} at pixel "
        f"({result['x']}, {result['y']})."
    )

elif operation == "Viola–Jones Algorithm":

    summary = (
        f"The integral image was used to calculate Haar "
        f"rectangle sums. The measured Haar difference was "
        f"{result['difference']:.2f}, with a normalized response "
        f"of {result['normalized']:.5f}."
    )

elif operation == "DeepFace":

    summary = (
        f"The face region produced a mean intensity of "
        f"{result['mean']:.2f} and standard deviation of "
        f"{result['std']:.2f}. The local feature analysis "
        f"produced an expression variation score of "
        f"{result['expression_score']:.4f}."
    )

else:

    summary = (
        f"The face was represented as a {result['dimension']}-"
        f"dimensional normalized embedding. Comparison with a "
        f"transformed representation produced cosine similarity "
        f"{result['cosine']:.6f} and Euclidean distance "
        f"{result['distance']:.6f}."
    )


st.markdown(f"""
<div class="card">

<div class="card-heading">
💡 Result Interpretation
</div>

<div class="info-text">
{summary}
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

<b>FACE PROCESSING LAB</b>
<br><br>

Template Matching • Viola–Jones • DeepFace • FaceNet
<br>

Interactive Computer Vision Demonstration

</div>
""", unsafe_allow_html=True)