import streamlit as st
import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLOv8 model
model_path = "ppe1trained-model.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO(model_path).to(device)  # Load model on available device

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Error: Unable to load image at {image_path}")
        return
    results = model(image)
    draw_boxes(image, results)
    st.image(image, channels="BGR")
    output_path = image_path.replace(".", "_output.")
    cv2.imwrite(output_path, image)
    st.success(f"Processing complete. Output saved at: {output_path}")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Unable to open video at {video_path}")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    output_path = video_path.replace(".", "_output.")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        draw_boxes(frame, results)
        out.write(frame)
    cap.release()
    out.release()
    st.success(f"Processing complete. Output saved at: {output_path}")

def process_cctv_feed(cctv_url):
    cap = cv2.VideoCapture(cctv_url)
    if not cap.isOpened():
        st.error(f"Error: Unable to open CCTV feed at {cctv_url}")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        draw_boxes(frame, results)
        st.image(frame, channels="BGR")
    cap.release()

def draw_boxes(image, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            label = model.names[int(box.cls[0].item())]
            # if label not in ["goggles", "no_goggles"]:  # Exclude goggles and no_goggles
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f"{label}: {confidence:.2f}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

st.title("PPE Detection")

input_type = st.selectbox("Select input type", ["Image", "Video", "CCTV Feed"])

if input_type == "Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        image_path = f"temp_{image_file.name}"
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())
        process_image(image_path)

elif input_type == "Video":
    video_file = st.file_uploader("Upload a video", type=["mp4"])
    if video_file is not None:
        video_path = f"temp_{video_file.name}"
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        process_video(video_path)

elif input_type == "CCTV Feed":
    cctv_url = st.text_input("Enter CCTV feed URL")
    if cctv_url:
        process_cctv_feed(cctv_url)