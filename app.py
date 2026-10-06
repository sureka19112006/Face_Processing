import os

# ============================================================
# DEPLOYMENT-SAFE ENVIRONMENT SETTINGS
# MUST COME BEFORE TENSORFLOW / DEEPFACE IMPORTS
# ============================================================

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "50000000")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time
import cv2


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

.section-title {
    margin-top: 30px;
    margin-bottom: 15px;
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    color: #ffd5c3;
    border-left: 5px solid #ff987c;
    padding-left: 15px;
}

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

.image-label {
    text-align: center;
    font-family: 'Playfair Display', serif;
    color: #ffd0bf;
    font-size: 20px;
    margin-bottom: 10px;
}

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

[data-testid="stFileUploader"] {
    background: rgba(30,18,30,0.75);
    border: 1px dashed rgba(255,154,126,0.5);
    border-radius: 16px;
    padding: 10px;
}

[data-baseweb="select"] > div {
    background-color: #17101a;
    border-color: rgba(255,154,126,0.35);
}

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

.warning-box {
    background: rgba(255, 170, 80, 0.08);
    border: 1px solid rgba(255, 170, 80, 0.3);
    border-radius: 15px;
    padding: 15px;
    color: #ffd5a8;
    margin: 15px 0;
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
# SAFE IMAGE SIZE
# ============================================================

MAX_PROCESSING_SIZE = 512


def resize_for_processing(img, max_size=MAX_PROCESSING_SIZE):

    img = img.convert("RGB")

    w, h = img.size

    scale = min(
        1.0,
        max_size / max(w, h)
    )

    if scale < 1:

        img = img.resize(
            (
                max(1, int(w * scale)),
                max(1, int(h * scale))
            ),
            Image.Resampling.LANCZOS
        )

    return img


# ============================================================
# IMAGE UTILITIES
# ============================================================

def load_default_image():

    try:

        from skimage import data

        arr = data.astronaut()

        return Image.fromarray(
            arr
        ).convert("RGB")

    except Exception:

        img = Image.new(
            "RGB",
            (512, 512),
            "#d8b09c"
        )

        draw = ImageDraw.Draw(img)

        draw.ellipse(
            (130, 90, 380, 390),
            fill="#d39b7c"
        )

        draw.ellipse(
            (190, 180, 220, 210),
            fill="#241c1b"
        )

        draw.ellipse(
            (290, 180, 320, 210),
            fill="#241c1b"
        )

        draw.arc(
            (200, 230, 310, 320),
            0,
            180,
            fill="#4b2525",
            width=8
        )

        return img


def pil_to_array(img):

    return np.array(
        img.convert("RGB")
    )


def gray_array(img):

    arr = pil_to_array(
        img
    ).astype(
        np.float32
    )

    return (
        0.299 * arr[:, :, 0]
        + 0.587 * arr[:, :, 1]
        + 0.114 * arr[:, :, 2]
    )


def pil_to_bgr(img):

    rgb = np.array(
        img.convert("RGB")
    )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def actual_template_matching(
    image,
    template
):

    image_gray = cv2.cvtColor(
        pil_to_bgr(image),
        cv2.COLOR_BGR2GRAY
    )

    template_gray = cv2.cvtColor(
        pil_to_bgr(template),
        cv2.COLOR_BGR2GRAY
    )

    ih, iw = image_gray.shape
    th, tw = template_gray.shape

    if th > ih or tw > iw:

        scale = min(
            (iw - 2) / tw,
            (ih - 2) / th
        )

        if scale <= 0:
            raise ValueError(
                "Template image is too large."
            )

        template_gray = cv2.resize(
            template_gray,
            (
                max(1, int(tw * scale)),
                max(1, int(th * scale))
            ),
            interpolation=cv2.INTER_AREA
        )

        th, tw = template_gray.shape

    result = cv2.matchTemplate(
        image_gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
        result
    )

    output = image.copy()

    draw = ImageDraw.Draw(
        output
    )

    x, y = max_loc

    draw.rectangle(
        (
            x,
            y,
            x + tw,
            y + th
        ),
        outline="#ff806f",
        width=5
    )

    draw.text(
        (
            x + 8,
            y + 8
        ),
        f"MATCH {max_val:.4f}",
        fill="#ffd0bf"
    )

    return output, {
        "score": float(max_val),
        "x": int(x),
        "y": int(y),
        "width": int(tw),
        "height": int(th)
    }


# ============================================================
# HAAR CASCADE
# ============================================================

@st.cache_resource
def load_haar_cascade():

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    cascade = cv2.CascadeClassifier(
        cascade_path
    )

    if cascade.empty():

        raise RuntimeError(
            "Haar Cascade could not be loaded."
        )

    return cascade


def viola_jones_detection(img):

    bgr = pil_to_bgr(
        img
    )

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    )

    cascade = load_haar_cascade()

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    output = img.copy()

    draw = ImageDraw.Draw(
        output
    )

    for i, (x, y, w, h) in enumerate(
        faces
    ):

        draw.rectangle(
            (
                int(x),
                int(y),
                int(x + w),
                int(y + h)
            ),
            outline="#ff806f",
            width=5
        )

        draw.text(
            (
                int(x + 8),
                int(y + 8)
            ),
            f"FACE {i + 1}",
            fill="#ffd0bf"
        )

    return output, faces


# ============================================================
# INTEGRAL IMAGE
# ============================================================

def integral_image(gray):

    return np.cumsum(
        np.cumsum(
            gray,
            axis=0
        ),
        axis=1
    )


def rectangle_sum(
    ii,
    x1,
    y1,
    x2,
    y2
):

    h, w = ii.shape

    x1 = max(
        0,
        min(w - 1, x1)
    )

    x2 = max(
        0,
        min(w - 1, x2)
    )

    y1 = max(
        0,
        min(h - 1, y1)
    )

    y2 = max(
        0,
        min(h - 1, y2)
    )

    total = ii[y2, x2]

    if x1 > 0:

        total -= ii[
            y2,
            x1 - 1
        ]

    if y1 > 0:

        total -= ii[
            y1 - 1,
            x2
        ]

    if x1 > 0 and y1 > 0:

        total += ii[
            y1 - 1,
            x1 - 1
        ]

    return float(total)


def viola_jones_analysis(img):

    gray = gray_array(
        img
    )

    ii = integral_image(
        gray
    )

    output, faces = viola_jones_detection(
        img
    )

    if len(faces) > 0:

        x, y, width, height = [
            int(v)
            for v in faces[0]
        ]

    else:

        h, w = gray.shape

        x = int(w * 0.2)
        y = int(h * 0.12)

        width = int(w * 0.6)
        height = int(h * 0.7)

    x2 = min(
        gray.shape[1],
        x + width
    )

    y2 = min(
        gray.shape[0],
        y + height
    )

    mid_x = x + (
        x2 - x
    ) // 2

    left_sum = rectangle_sum(
        ii,
        x,
        y,
        mid_x - 1,
        y2 - 1
    )

    right_sum = rectangle_sum(
        ii,
        mid_x,
        y,
        x2 - 1,
        y2 - 1
    )

    haar_difference = (
        left_sum -
        right_sum
    )

    face_area = max(
        1,
        (x2 - x) *
        (y2 - y)
    )

    normalized_difference = (
        haar_difference /
        face_area
    )

    draw = ImageDraw.Draw(
        output
    )

    draw.line(
        (
            mid_x,
            y,
            mid_x,
            y2
        ),
        fill="#c28bd1",
        width=4
    )

    draw.text(
        (
            x + 10,
            y + 10
        ),
        "HAAR FEATURE",
        fill="#ffd0bf"
    )

    return output, {
        "faces": len(faces),
        "left_sum": left_sum,
        "right_sum": right_sum,
        "difference": haar_difference,
        "normalized": normalized_difference,
        "area": face_area,
        "integral_center": float(
            ii[y, x]
        )
    }


# ============================================================
# DEEPFACE - LAZY IMPORT
# ============================================================

@st.cache_resource
def load_deepface():

    try:

        import tensorflow as tf

        try:

            tf.config.set_visible_devices(
                [],
                "GPU"
            )

        except Exception:
            pass

        try:

            tf.config.threading.set_intra_op_parallelism_threads(
                1
            )

            tf.config.threading.set_inter_op_parallelism_threads(
                1
            )

        except Exception:
            pass

        from deepface import DeepFace

        return DeepFace

    except Exception as e:

        raise RuntimeError(
            "DeepFace could not be loaded. "
            "Deployment-safe DeepFace initialization failed: "
            + str(e)
        )


# ============================================================
# DEPLOYMENT-SAFE DEEPFACE
# ============================================================

def deepface_analysis(img):

    start = time.time()

    DeepFace = load_deepface()

    safe_img = resize_for_processing(
        img,
        384
    )

    bgr = pil_to_bgr(
        safe_img
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # ONLY EMOTION MODEL IS USED.
    #
    # Age / Gender / Race models are NOT loaded.
    # This reduces deployment memory usage.
    # --------------------------------------------------------

    analysis = DeepFace.analyze(
        img_path=bgr,
        actions=["emotion"],
        detector_backend="opencv",
        enforce_detection=False,
        align=True,
        silent=True
    )

    if isinstance(
        analysis,
        dict
    ):

        analysis = [
            analysis
        ]

    if len(analysis) == 0:

        raise RuntimeError(
            "DeepFace did not return an analysis result."
        )

    first = analysis[0]

    output = safe_img.copy()

    draw = ImageDraw.Draw(
        output
    )

    face_region = first.get(
        "region",
        {}
    )

    if face_region:

        x = int(
            face_region.get(
                "x",
                0
            )
        )

        y = int(
            face_region.get(
                "y",
                0
            )
        )

        w = int(
            face_region.get(
                "w",
                0
            )
        )

        h = int(
            face_region.get(
                "h",
                0
            )
        )

        if w > 0 and h > 0:

            draw.rectangle(
                (
                    x,
                    y,
                    x + w,
                    y + h
                ),
                outline="#ff806f",
                width=5
            )

            draw.text(
                (
                    x + 8,
                    y + 8
                ),
                "DEEPFACE",
                fill="#ffd0bf"
            )

    emotion = str(
        first.get(
            "dominant_emotion",
            "Unknown"
        )
    )

    emotion_scores = first.get(
        "emotion",
        {}
    )

    if isinstance(
        emotion_scores,
        dict
    ) and emotion_scores:

        expression_score = float(
            max(
                emotion_scores.values()
            ) / 100.0
        )

    else:

        expression_score = 0.0

    gray = gray_array(
        safe_img
    )

    mean = float(
        np.mean(gray)
    )

    std = float(
        np.std(gray)
    )

    processing_time = (
        time.time()
        -
        start
    )

    return output, {

        "faces": len(
            analysis
        ),

        "emotion": emotion,

        "confidence":
            float(
                first.get(
                    "face_confidence",
                    0
                ) or 0
            ),

        "expression_score":
            expression_score,

        "expression":
            emotion,

        "mean":
            mean,

        "std":
            std,

        "processing_time":
            processing_time
    }


# ============================================================
# FACENET - LAZY IMPORT
# ============================================================

@st.cache_resource
def load_facenet_model():

    try:

        DeepFace = load_deepface()

        model = DeepFace.build_model(
            "Facenet"
        )

        return DeepFace, model

    except Exception as e:

        raise RuntimeError(
            "FaceNet model could not be loaded: "
            + str(e)
        )


# ============================================================
# FACENET EMBEDDING
# ============================================================

def get_facenet_embedding(img):

    DeepFace, model = load_facenet_model()

    safe_img = resize_for_processing(
        img,
        384
    )

    bgr = pil_to_bgr(
        safe_img
    )

    embedding_objects = DeepFace.represent(
        img_path=bgr,
        model_name="Facenet",
        detector_backend="opencv",
        enforce_detection=False,
        align=True,
        normalization="base"
    )

    if isinstance(
        embedding_objects,
        dict
    ):

        embedding_objects = [
            embedding_objects
        ]

    if len(
        embedding_objects
    ) == 0:

        raise RuntimeError(
            "No FaceNet embedding was generated."
        )

    embedding = np.array(
        embedding_objects[0]["embedding"],
        dtype=np.float32
    )

    return embedding


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:

        return 0.0

    return float(
        np.dot(a, b)
        /
        denominator
    )


# ============================================================
# FACENET ANALYSIS
# ============================================================

def facenet_analysis(
    img,
    comparison_image
):

    start = time.time()

    embedding = get_facenet_embedding(
        img
    )

    embedding2 = get_facenet_embedding(
        comparison_image
    )

    cosine = cosine_similarity(
        embedding,
        embedding2
    )

    distance = float(
        np.linalg.norm(
            embedding -
            embedding2
        )
    )

    verification_threshold = 0.70

    verified = (
        cosine >=
        verification_threshold
    )

    output = resize_for_processing(
        img,
        384
    ).copy()

    draw = ImageDraw.Draw(
        output
    )

    draw.rectangle(
        (
            5,
            5,
            output.width - 5,
            output.height - 5
        ),
        outline="#ff806f",
        width=4
    )

    draw.text(
        (
            15,
            15
        ),
        "FACENET EMBEDDING",
        fill="#ffd0bf"
    )

    elapsed = (
        time.time()
        -
        start
    )

    return output, {

        "dimension":
            len(embedding),

        "cosine":
            cosine,

        "distance":
            distance,

        "first":
            embedding[:8],

        "mean":
            float(
                np.mean(
                    embedding
                )
            ),

        "std":
            float(
                np.std(
                    embedding
                )
            ),

        "verified":
            bool(
                verified
            ),

        "threshold":
            verification_threshold,

        "processing_time":
            elapsed
    }


# ============================================================
# PIXEL LENS
# ============================================================

def pixel_lens(
    img,
    px,
    py
):

    gray = gray_array(
        img
    )

    h, w = gray.shape

    px = max(
        0,
        min(
            w - 1,
            px
        )
    )

    py = max(
        0,
        min(
            h - 1,
            py
        )
    )

    half = 4

    x1 = max(
        0,
        px - half
    )

    x2 = min(
        w,
        px + half
    )

    y1 = max(
        0,
        py - half
    )

    y2 = min(
        h,
        py + half
    )

    matrix = gray[
        y1:y2,
        x1:x2
    ]

    return {

        "matrix":
            matrix,

        "center":
            float(
                gray[py, px]
            ),

        "mean":
            float(
                np.mean(matrix)
            ),

        "minimum":
            float(
                np.min(matrix)
            ),

        "maximum":
            float(
                np.max(matrix)
            ),

        "range":
            float(
                np.max(matrix)
                -
                np.min(matrix)
            ),

        "x":
            px,

        "y":
            py
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
    """,
    unsafe_allow_html=True)

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
    """,
    unsafe_allow_html=True)


if uploaded is not None:

    original = Image.open(
        uploaded
    ).convert("RGB")

else:

    original = load_default_image()


original = resize_for_processing(
    original,
    MAX_PROCESSING_SIZE
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


# ============================================================
# EXTRA INPUTS
# ============================================================

template_image = None
comparison_image = None


if operation == "Template Matching":

    st.markdown("""
    <div class="card">

    <div class="card-heading">
    🖼️ Template Image
    </div>

    <div class="info-text">
    Upload a separate small image/template.
    The application will search for this exact
    visual pattern inside the input image.
    </div>

    </div>
    """,
    unsafe_allow_html=True)

    template_uploaded = st.file_uploader(
        "Upload template image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="template_uploader"
    )

    if template_uploaded is not None:

        template_image = Image.open(
            template_uploaded
        ).convert("RGB")

        template_image = resize_for_processing(
            template_image,
            300
        )

        st.image(
            template_image,
            caption="Selected Template",
            width=250
        )


elif operation == "FaceNet":

    st.markdown("""
    <div class="card">

    <div class="card-heading">
    🧬 FaceNet Comparison Image
    </div>

    <div class="info-text">
    Upload another face image. FaceNet generates
    embeddings for both images and performs
    actual face comparison.
    </div>

    </div>
    """,
    unsafe_allow_html=True)

    comparison_uploaded = st.file_uploader(
        "Upload comparison face image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="comparison_uploader"
    )

    if comparison_uploaded is not None:

        comparison_image = Image.open(
            comparison_uploaded
        ).convert("RGB")

        comparison_image = resize_for_processing(
            comparison_image,
            MAX_PROCESSING_SIZE
        )

        st.image(
            comparison_image,
            caption="FaceNet Comparison Image",
            width=300
        )


# ============================================================
# OPERATION INFO
# ============================================================

operation_info = {

    "Template Matching":
        "Finds the location where a selected template best matches the image.",

    "Viola–Jones Algorithm":
        "Uses Haar-like features, integral images and a Haar Cascade classifier for face detection.",

    "DeepFace":
        "Uses an actual DeepFace deep-learning model for facial emotion analysis.",

    "FaceNet":
        "Generates actual FaceNet embeddings and compares two faces using vector similarity."
}


st.markdown(f"""
<div class="operation-card">

<h3>⚙️ {operation}</h3>

<p>
{operation_info[operation]}
</p>

</div>
""",
unsafe_allow_html=True)


# ============================================================
# DEPLOYMENT NOTE FOR DEEPFACE
# ============================================================

if operation == "DeepFace":

    st.markdown("""
    <div class="warning-box">

    <b>Deployment-safe DeepFace mode</b><br><br>

    DeepFace uses the actual deep-learning emotion model.
    The emotion model is loaded only when DeepFace
    is selected, helping reduce deployment memory usage.

    </div>
    """,
    unsafe_allow_html=True)


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
1. Convert the input and template images into grayscale.
2. Use the uploaded template as the reference.
3. Slide the template across the input image.
4. Calculate normalized correlation using OpenCV.
5. Find the highest matching position.
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
3. Load the Haar Cascade classifier.
4. Detect faces using the trained cascade.
5. Calculate Haar rectangle sums.
6. Display the detected face regions.
"""
},

"DeepFace": {

"definition":
"""
DeepFace is a deep-learning based face-analysis framework.
This deployment-safe version performs actual facial emotion
analysis using the DeepFace emotion model.
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
1. Obtain the input face image.
2. Detect and align the face.
3. Pass the face through the actual DeepFace emotion model.
4. Predict the dominant facial emotion.
5. Calculate image statistics.
6. Display the actual analysis result.
"""
},

"FaceNet": {

"definition":
"""
FaceNet represents a face as a numerical embedding vector.
The actual FaceNet model generates a face representation
which can be compared using vector distance and similarity.
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
1. Detect and align the face.
2. Generate the actual FaceNet embedding.
3. Generate an embedding for the comparison image.
4. Calculate cosine similarity.
5. Calculate Euclidean distance.
6. Perform face verification using similarity threshold.
"""
}

}

t = theory[
    operation
]

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
""",
unsafe_allow_html=True)


# ============================================================
# PROCESS IMAGE
# ============================================================

processing_error = None

output = original.copy()

result = {}


try:

    if operation == "Template Matching":

        if template_image is not None:

            output, result = actual_template_matching(
                original,
                template_image
            )

        else:

            st.info(
                "Upload a template image to perform actual template matching."
            )

            result = {
                "score": 0.0,
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0
            }


    elif operation == "Viola–Jones Algorithm":

        output, result = viola_jones_analysis(
            original
        )


    elif operation == "DeepFace":

        output, result = deepface_analysis(
            original
        )


    elif operation == "FaceNet":

        if comparison_image is not None:

            output, result = facenet_analysis(
                original,
                comparison_image
            )

        else:

            st.info(
                "Upload a comparison image to perform actual FaceNet comparison."
            )

            result = {
                "dimension": 128,
                "cosine": 0.0,
                "distance": 0.0,
                "first": np.zeros(8),
                "mean": 0.0,
                "std": 0.0,
                "verified": False,
                "threshold": 0.70,
                "processing_time": 0.0
            }


except Exception as e:

    processing_error = str(e)

    output = original.copy()

    result = {}


if processing_error:

    st.error(
        "Processing error: "
        + processing_error
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


def calc_card(
    title,
    value,
    formula
):

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
    """,
    unsafe_allow_html=True)


# ============================================================
# TEMPLATE CALCULATIONS
# ============================================================

if operation == "Template Matching":

    c1, c2, c3 = st.columns(3)

    with c1:

        calc_card(
            "Correlation Score",
            f"{result.get('score', 0):.4f}",
            "OpenCV normalized cross-correlation"
        )

    with c2:

        calc_card(
            "Best X Position",
            f"{result.get('x', 0)} px",
            "Horizontal template position"
        )

    with c3:

        calc_card(
            "Best Y Position",
            f"{result.get('y', 0)} px",
            "Vertical template position"
        )

    c4, c5 = st.columns(2)

    with c4:

        calc_card(
            "Template Width",
            f"{result.get('width', 0)} px",
            "Uploaded template width"
        )

    with c5:

        calc_card(
            "Template Height",
            f"{result.get('height', 0)} px",
            "Uploaded template height"
        )


# ============================================================
# VIOLA JONES CALCULATIONS
# ============================================================

elif operation == "Viola–Jones Algorithm":

    c1, c2, c3 = st.columns(3)

    with c1:

        calc_card(
            "Detected Face Regions",
            str(
                result.get(
                    "faces",
                    0
                )
            ),
            "Actual Haar Cascade detections"
        )

    with c2:

        calc_card(
            "White Region Sum",
            f"{result.get('left_sum', 0):.2f}",
            "Integral-image rectangle sum"
        )

    with c3:

        calc_card(
            "Black Region Sum",
            f"{result.get('right_sum', 0):.2f}",
            "Integral-image rectangle sum"
        )

    c4, c5 = st.columns(2)

    with c4:

        calc_card(
            "Haar Difference",
            f"{result.get('difference', 0):.2f}",
            "White − Black"
        )

    with c5:

        calc_card(
            "Normalized Haar Response",
            f"{result.get('normalized', 0):.5f}",
            "Haar difference / face area"
        )


# ============================================================
# DEEPFACE CALCULATIONS
# ============================================================

elif operation == "DeepFace":

    c1, c2, c3 = st.columns(3)

    with c1:

        calc_card(
            "Detected Face Regions",
            str(
                result.get(
                    "faces",
                    0
                )
            ),
            "Actual DeepFace face detection"
        )

    with c2:

        calc_card(
            "Emotion",
            str(
                result.get(
                    "emotion",
                    "Unknown"
                )
            ),
            "Actual DeepFace emotion model"
        )

    with c3:

        calc_card(
            "Face Confidence",
            f"{result.get('confidence', 0):.2f}",
            "DeepFace detector confidence"
        )

    c4, c5, c6 = st.columns(3)

    with c4:

        calc_card(
            "Mean Intensity",
            f"{result.get('mean', 0):.4f}",
            "μ = Σxᵢ / N"
        )

    with c5:

        calc_card(
            "Std. Deviation",
            f"{result.get('std', 0):.4f}",
            "σ = √[Σ(xᵢ−μ)²/N]"
        )

    with c6:

        calc_card(
            "Processing Time",
            f"{result.get('processing_time', 0):.5f} s",
            "End time − start time"
        )

    st.markdown(f"""
    <div class="card">

    <div class="card-heading">
    🙂 Feature Interpretation
    </div>

    <div class="info-text">

    Dominant Emotion:
    {result.get("emotion", "Unknown")}

    </div>

    </div>
    """,
    unsafe_allow_html=True)


# ============================================================
# FACENET CALCULATIONS
# ============================================================

else:

    c1, c2, c3 = st.columns(3)

    with c1:

        calc_card(
            "Embedding Dimension",
            str(
                result.get(
                    "dimension",
                    128
                )
            ),
            "Actual FaceNet embedding dimension"
        )

    with c2:

        calc_card(
            "Cosine Similarity",
            f"{result.get('cosine', 0):.6f}",
            "(A · B) / (||A|| ||B||)"
        )

    with c3:

        calc_card(
            "Euclidean Distance",
            f"{result.get('distance', 0):.6f}",
            "√Σ(Aᵢ − Bᵢ)²"
        )

    c4, c5 = st.columns(2)

    with c4:

        calc_card(
            "Face Verification",
            (
                "MATCH"
                if result.get(
                    "verified",
                    False
                )
                else "NOT MATCHED"
            ),
            "Actual FaceNet cosine-similarity verification"
        )

    with c5:

        threshold = result.get(
            "threshold",
            None
        )

        threshold_text = (
            f"{threshold:.6f}"
            if isinstance(
                threshold,
                (float, int)
            )
            else "Model default"
        )

        calc_card(
            "Verification Threshold",
            threshold_text,
            "FaceNet cosine similarity threshold"
        )

    c6, c7 = st.columns(2)

    with c6:

        calc_card(
            "Embedding Mean",
            f"{result.get('mean', 0):.6f}",
            "Mean of actual FaceNet embedding"
        )

    with c7:

        calc_card(
            "Embedding Std.",
            f"{result.get('std', 0):.6f}",
            "Standard deviation of embedding"
        )

    st.markdown("""
    <div class="card">

    <div class="card-heading">
    🧬 Embedding Preview
    </div>

    <div class="info-text">
    First 8 dimensions of the actual FaceNet representation:
    </div>

    <div class="formula">
    """,
    unsafe_allow_html=True)

    embedding_preview = result.get(
        "first",
        np.zeros(8)
    )

    st.markdown(
        " &nbsp;&nbsp; ".join(
            f"{v:.4f}"
            for v in embedding_preview
        )
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# PIXEL LENS
# ============================================================

st.markdown(
    '<div class="section-title">06 • PIXEL LENS</div>',
    unsafe_allow_html=True
)

gray = gray_array(
    original
)

h, w = gray.shape

px_col, py_col = st.columns(2)

with px_col:

    px = st.slider(
        "Select X pixel",
        0,
        max(
            0,
            w - 1
        ),
        w // 2
    )

with py_col:

    py = st.slider(
        "Select Y pixel",
        0,
        max(
            0,
            h - 1
        ),
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
""",
unsafe_allow_html=True)


matrix = pixel["matrix"]

matrix_html = (
    '<div class="pixel-box">'
)

for row in matrix:

    for value in row:

        intensity = int(
            max(
                0,
                min(
                    255,
                    value
                )
            )
        )

        text_color = (
            "#111111"
            if intensity > 160
            else "#ffffff"
        )

        matrix_html += (
            f'<div class="pixel-value" '
            f'style="background:rgb('
            f'{intensity},{intensity},{intensity});'
            f'color:{text_color};">'
            f'{intensity}'
            f'</div>'
        )

matrix_html += (
    '</div>'
)

st.markdown(
    matrix_html,
    unsafe_allow_html=True
)


# ============================================================
# PROCESSING SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">07 • PROCESSING SUMMARY</div>',
    unsafe_allow_html=True
)


if operation == "Template Matching":

    summary = (
        f"The uploaded template was compared against "
        f"the input image using OpenCV normalized "
        f"template matching. The highest correlation "
        f"was {result.get('score', 0):.4f} at pixel "
        f"({result.get('x', 0)}, "
        f"{result.get('y', 0)})."
    )


elif operation == "Viola–Jones Algorithm":

    summary = (
        f"The Haar Cascade classifier detected "
        f"{result.get('faces', 0)} face region(s). "
        f"The integral image was used for rectangle "
        f"sum calculations. The Haar difference was "
        f"{result.get('difference', 0):.2f}."
    )


elif operation == "DeepFace":

    summary = (
        f"DeepFace performed actual emotion analysis "
        f"and detected "
        f"{result.get('faces', 0)} face region(s). "
        f"The dominant emotion was "
        f"{result.get('emotion', 'Unknown')}."
    )


else:

    summary = (
        f"The actual FaceNet model generated a "
        f"{result.get('dimension', 128)}-dimensional "
        f"face embedding. The two face embeddings "
        f"produced cosine similarity "
        f"{result.get('cosine', 0):.6f} and "
        f"Euclidean distance "
        f"{result.get('distance', 0):.6f}. "
        f"Verification result: "
        f"{'MATCH' if result.get('verified', False) else 'NOT MATCHED'}."
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
""",
unsafe_allow_html=True)


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
