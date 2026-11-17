"""Simple local webcam check. Press Q in the video window to exit."""
import cv2

def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit(
            "Could not access webcam. Check Windows camera permissions, "
            "close other camera apps, or try camera index 1."
        )
    print("PharmaSight AI webcam test started. Press Q in the video window to exit.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a webcam frame.")
                break
            cv2.putText(
                frame, "PharmaSight AI - Camera Test", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow("PharmaSight AI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
