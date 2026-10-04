import os
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")  # safety net if a GUI build sneaks back in


import cv2


def show(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video's frame rate to calculate proper delay
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30  # delay in milliseconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame in an OpenCV window
        cv2.imshow('Video Player', frame)

        # Wait for the frame duration, and allow the user to press 'q' to quit early
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Clean up windows and release memory
    cap.release()
    cv2.destroyAllWindows()