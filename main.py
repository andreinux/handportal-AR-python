
import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "hand_landmarker.task"
PORTAL_W = 400
PORTAL_H = 300

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
Options = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = Options(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)


def apply_effect(image, mode):
    # MODE 1: NIGHT VISION
    if mode == 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Boost contrast
        gray = cv2.equalizeHist(gray)

        green = np.zeros_like(image)
        green[:, :, 1] = gray
        green[:, :, 2] = (gray * 0.22).astype(np.uint8)

        # Scanlines
        green[::4, :] = (green[::4, :] * 0.65).astype(np.uint8)

        # Crosshair
        h, w = gray.shape
        cv2.line(green, (w // 2 - 15, h // 2),
                 (w // 2 + 15, h // 2), (0, 180, 0), 1)
        cv2.line(green, (w // 2, h // 2 - 15),
                 (w // 2, h // 2 + 15), (0, 180, 0), 1)

        return green

    # MODE 2: THERMAL VISION
    elif mode == 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    # MODE 3: CYBERPUNK GLITCH
    elif mode == 3:
        b, g, r = cv2.split(image)

        # Shift color channels in different directions
        shift = np.random.randint(-8, 9)

        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)

        result = cv2.merge([b, g, r])

        # Occasional digital tearing
        if np.random.random() < 0.12:
            y = np.random.randint(0, PORTAL_H - 15)
            height = np.random.randint(3, 15)
            offset = np.random.randint(-25, 26)

            result[y:y + height] = np.roll(
                result[y:y + height], offset, axis=1
            )

        # Subtle scanlines
        result[::4, :] = (
            result[::4, :] * 0.75
        ).astype(np.uint8)

        return result

    # MODE 4: KALEIDOSCOPE
    elif mode == 4:
        h, w = image.shape[:2]
        half_w = w // 2
        half_h = h // 2

        small = cv2.resize(
            image, (half_w, half_h)
        )

        result = np.zeros_like(image)

        # Top-left quadrant
        result[:half_h, :half_w] = small

        # Top-right: mirrored horizontally
        result[:half_h, half_w:] = cv2.flip(
            small, 1
        )

        # Bottom-left: mirrored vertically
        result[half_h:, :half_w] = cv2.flip(
            small, 0
        )

        # Bottom-right: mirrored both ways
        result[half_h:, half_w:] = cv2.flip(
            small, -1
        )

        return result

    return image


effect_names = {
    1: "NIGHT VISION",
    2: "THERMAL VISION",
    3: "CYBERPUNK GLITCH",
    4: "KALEIDOSCOPE"
}

mode = 1
camera = cv2.VideoCapture(0)

try:
    with HandLandmarker.create_from_options(options) as tracker:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not access webcam.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                cv2.getTickCount() * 1000
                / cv2.getTickFrequency()
            )

            results = tracker.detect_for_video(
                image, timestamp
            )

            if (
                results.hand_landmarks
                and len(results.hand_landmarks) == 2
            ):
                hand_points = []

                for hand in results.hand_landmarks:
                    index_tip = hand[8]
                    thumb_tip = hand[4]

                    hand_points.append([
                        (
                            int(index_tip.x * width),
                            int(index_tip.y * height)
                        ),
                        (
                            int(thumb_tip.x * width),
                            int(thumb_tip.y * height)
                        )
                    ])

                corners = np.float32([
                    hand_points[0][0],
                    hand_points[1][0],
                    hand_points[1][1],
                    hand_points[0][1]
                ])

                destination = np.float32([
                    [0, 0],
                    [PORTAL_W - 1, 0],
                    [PORTAL_W - 1, PORTAL_H - 1],
                    [0, PORTAL_H - 1]
                ])

                matrix = cv2.getPerspectiveTransform(
                    corners, destination
                )

                # Extract the view inside the portal
                portal = cv2.warpPerspective(
                    frame,
                    matrix,
                    (PORTAL_W, PORTAL_H)
                )

                # Apply the selected visual effect
                effect = apply_effect(portal, mode)

                reverse_matrix = cv2.getPerspectiveTransform(
                    destination, corners
                )

                warped = cv2.warpPerspective(
                    effect,
                    reverse_matrix,
                    (width, height)
                )

                mask = np.zeros(
                    (height, width), dtype=np.uint8
                )

                cv2.fillConvexPoly(
                    mask, corners.astype(np.int32), 255
                )

                frame[mask == 255] = warped[mask == 255]

                # Glowing-style green portal border
                cv2.polylines(
                    frame,
                    [corners.astype(np.int32)],
                    True,
                    (0, 255, 0),
                    3
                )

           


      
            

            cv2.imshow("HAND PORTAL - MULTI EFFECTS", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                mode = int(chr(key))

finally:
    camera.release()
    cv2.destroyAllWindows()
