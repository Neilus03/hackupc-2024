import cv2
from inference_sdk import InferenceHTTPClient
import base64

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="RQyd3bogNb9shYvNLGqO"
)

def detect_hand_gesture_webcam():
    cap = cv2.VideoCapture(0)  # Webcam capture
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        # Convert buffer to bytes
        img_bytes = buffer.tobytes()
        # Encode image bytes in Base64 format
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        # Perform inference
        result = CLIENT.infer(img_base64, model_id="hand-gesture-r7qgb/2")
        # Process the inference result
        # (This depends on the format of the result returned by the inference API)
        # Display the frame
        cv2.imshow('Hand Gesture Detection', frame)
        '''
        if result['predictions'] != []:
            for i in range(len(result['predictions'])):
              resultt = result['predictions'][i]['class']
              print(resultt)
        '''
        
        if cv2.waitKey(1) == ord('q'):  # Quit on 'q' key press
            break
    cap.release()
    cv2.destroyAllWindows()

detect_hand_gesture_webcam()

