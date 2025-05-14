# smart-emotion-detector-ai
AI-Emotion-Detector: Real-time Facial Expression Recognition with Python &amp; Deep Learning

If You Install Fresh then follow these Process - 

Installation & Usage Guide

1. System Requirements

1. OS: Windows or Linux
2. Python: Version 3.10 or above
3. Hardware: Webcam (built-in or USB), Minimum 4GB RAM



2. Required Python Libraries**

Install all required libraries using pip:

1. "pip install opencv-python keras tensorflow numpy h5py"
Optional for GUI-less systems:
1. "pip install opencv-python-headless"


3. Project Folder Structure**

AI-Emotions-Detection/
├── check_emotions.py
├── SmartEmotionModel_v1.h5
├── AI Smart_face_detector.xml
├── myenv/ (your virtual environment)


4. Setup Virtual Environment (Recommended) =
   
1. python -m venv myenv
2. myenv\Scripts\activate      #On Windows
3. pip install -r requirements.txt  # Optional, if available


5. Run the Application -

Activate the virtual environment and run:
1. python check_emotions.py
2. The webcam window will open showing real-time emotion detection.


6. How to Use

1. Look directly into the webcam.
2. The app detects your face and displays:

  - Emotion label (e.g., Happy, Sad, Angry)
  - Confidence percentage
  - Colored overlay for each emotion
  - Press `a` (as defined) to exit.






---------------------------------------------------------------- THANK YOU ------------------------------------------------------------------------



