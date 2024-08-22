import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st
import subprocess


st.set_page_config(layout="wide")
st.image('Gestures.jpeg')

col1,col2 = st.columns([2,1])
with col1:
    run = st.checkbox('Run',value ="True")
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")
genai.configure(api_key="AIzaSyDe2vQPA9Ykemvn2fw7HgQdJ2v8i2gOlpU")
model = genai.GenerativeModel('gemini-1.5-flash')


# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 380)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5 , minTrackCon=0.5)

def getHandInfo(img):
    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=False, flipType=True)

    # Check if any hands are detected
    if hands:
        hand1 = hands[0]  # Get the first hand detected
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand

        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand1)
        return fingers, lmList1
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList1 = info
    current_pos = None
    
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList1[8][0:2]  # Get the x, y coordinates of the index finger tip
        if prev_pos is None: 
            prev_pos = current_pos
        # Draw a line on the canvas
        cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (255, 0, 255), 10)
    elif fingers == [1,1,1,1,1]:
        canvas = np.zeros_like(img)

    
    return current_pos, canvas

def sendToAI(model, canvas,fingers):
    if fingers == [1,1,1,1,0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve the Math Problem ", pil_image])
        return response.text


prev_pos = None
canvas = None
output_text = None

# Continuously get frames from the webcam
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)

    if info:
        fingers, lmlist1 = info
        print(fingers)
        prev_pos, canvas = draw(info, prev_pos, canvas)  # Correctly unpack the values
        output_text = sendToAI(model,canvas,fingers)


    image_combines = cv2.addWeighted(img, 0.65, canvas, 0.35, 0)

    FRAME_WINDOW.image(image_combines,channels="BGR")
    

    if output_text:
        output_text_area.text(output_text)
    # Display the image in a window
    # cv2.imshow("Image", img)
    # cv2.imshow("Canvas", canvas)
    # cv2.imshow("image_combines", image_combines)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(2)
