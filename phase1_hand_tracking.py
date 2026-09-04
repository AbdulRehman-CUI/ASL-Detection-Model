"""Phase 1: real-time hand landmark tracking with OpenCV and MediaPipe."""

try:
    import cv2
    import mediapipe as mp
except ImportError as error:
    raise SystemExit(
        "Missing dependency. Install the project dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from error


WINDOW_TITLE = "Sign Language Translator - Phase 1"
CAMERA_INDEX = 0


def main():
    """Open the webcam, track hands, and display the annotated video."""
    hands_module = mp.solutions.hands
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print(
            f"Error: Could not open webcam at camera index {CAMERA_INDEX}. "
            "Check that it is connected and available to this application."
        )
        camera.release()
        return

    try:
        with hands_module.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            while True:
                frame_read, frame = camera.read()

                if not frame_read:
                    print("Error: Could not read a frame from the webcam.")
                    break

                # Mirror the feed, then convert BGR to RGB for MediaPipe.
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        drawing_utils.draw_landmarks(
                            frame,
                            hand_landmarks,
                            hands_module.HAND_CONNECTIONS,
                            drawing_styles.get_default_hand_landmarks_style(),
                            drawing_styles.get_default_hand_connections_style(),
                        )

                cv2.putText(
                    frame,
                    "Press Q or ESC to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_TITLE, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()