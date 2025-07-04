
# 🦺 PPE Detection using YOLOv8 and Streamlit

This application detects **Personal Protective Equipment (PPE)** such as helmets, vests, gloves, goggles, etc., from **images**, **videos**, or **live CCTV feeds** using a **custom-trained YOLOv8 model**. It provides a user-friendly interface built with **Streamlit** for seamless interaction.

---

## 📌 Features

- ✅ **Image and Video Detection**  
  Upload images or videos and automatically detect PPE items.

- 📷 **Live CCTV Feed Support**  
  Perform real-time detection via RTSP or HTTP CCTV stream URLs.

- 🧠 **Custom YOLOv8 Model**  
  Uses a fine-tuned model (`ppe1trained-model.pt`) specifically trained for PPE classes like helmet, vest, gloves, goggles, etc.

- 🔲 **Bounding Boxes with Labels**  
  Displays detection results with class names and confidence scores overlaid on the frame.

- 🖼️ **Output Preview and Storage**  
  Processed images and videos are displayed in the UI and saved locally with annotated results.

- 💻 **Optimized for GPU or CPU**  
  Automatically detects hardware and runs on GPU (`cuda`) if available, otherwise defaults to CPU.

---

## 📁 Project Structure


