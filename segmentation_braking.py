import cv2
import csv
import numpy as np
from pathlib import Path

def load_semantic_colors(csv_file):
    semantic_colors = {}

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["semantic_class"].strip().lower()

            rgb_str = row["rgb_values"].strip("[]")
            rgb = [int(v) for v in rgb_str.split(",")]

            if len(rgb) != 3:
                continue

            # Convert RGB → BGR (OpenCV)
            semantic_colors[name] = np.array(
                [rgb[2], rgb[1], rgb[0]], dtype=np.uint8
            )

    return semantic_colors

csv_path = Path(r'D:\school\kbs\classes_rgb_values.csv')
SEMANTIC_COLORS = load_semantic_colors(csv_path)

# =========================================================
# Paths
# =========================================================
frames_root = Path(r'D:\school\kbs\images')
labels_root = Path(r'D:\school\kbs\labels')

ROAD_COLOR = SEMANTIC_COLORS["road"]
ROADLINE_COLOR = SEMANTIC_COLORS["road line"]
SIDEWALK_COLOR = SEMANTIC_COLORS["sidewalk"]

IMAGE_WIDTH, IMAGE_HEIGHT = 1620, 1200

# =========================================================
# Thresholds (tuneable)
# =========================================================
CAR_AREA_THRESHOLD = 0.03
APPROACH_SPEED_THRESHOLD = 0.005

# =========================================================
# Ego-lane ROI (mid-low, centered)
# =========================================================
def ego_lane_roi(h, w):
    return (
        int(h * 0.65), int(h * 0.99),   # y1, y2
        int(w * 0.4), int(w * 0.6)    # x1, x2
    )

# =========================================================
# CAR danger: distance + approach speed
# =========================================================
def car_danger(label_img, prev_ratio):
    h, w, _ = label_img.shape
    y1, y2, x1, x2 = ego_lane_roi(h, w)

    roi = label_img[y1:y2, x1:x2]
    car_color = SEMANTIC_COLORS["car"]

    car_mask = np.all(roi == car_color, axis=2)
    car_ratio = np.sum(car_mask) / car_mask.size

    too_close = car_ratio > CAR_AREA_THRESHOLD
    approaching = False

    if prev_ratio is not None:
        approaching = (car_ratio - prev_ratio) > APPROACH_SPEED_THRESHOLD

    danger = too_close or approaching
    return danger, car_ratio, (x1, y1, x2, y2)

# =========================================================
# PEDESTRIAN danger: strict & close
# =========================================================

def pedestrian_roi(h, w):
    return (
        int(h * 0.60), int(h * 0.85),   # y1, y2 (very close)
        int(w * 0.45), int(w * 0.55)    # x1, x2 (center)
    )


def pedestrian_danger(label_img):
    h, w, _ = label_img.shape
    ped_color = SEMANTIC_COLORS["pedestrian"]

    y1, y2, x1, x2 = pedestrian_roi(h, w)
    roi = label_img[y1:y2, x1:x2]

    ped_mask = np.all(roi == ped_color, axis=2)
    danger = np.any(ped_mask)
    return danger, (x1, y1, x2, y2)

# =========================================================
# Oncoming traffic ROI (LEFT TURN DANGER ZONE)
# =========================================================
def oncoming_traffic_roi(h, w):
    return (
        int(h * 0.45), int(h * 0.70),   # y1, y2 (ahead)
        int(w * 0.15), int(w * 0.45)    # x1, x2 (left side)
    )

ONCOMING_CAR_THRESHOLD = 0.01  

def separated_by_non_road(label_img, ego_roi, target_roi):
    ey1, ey2, ex1, ex2 = ego_roi
    ty1, ty2, tx1, tx2 = target_roi

    mid_y1 = min(ey1, ty1)
    mid_y2 = max(ey2, ty2)
    mid_x1 = min(ex1, tx1)
    mid_x2 = max(ex2, tx2)

    if mid_y2 <= mid_y1 or mid_x2 <= mid_x1:
        return False

    band = label_img[mid_y1:mid_y2, mid_x1:mid_x2]

    road_mask = np.all(band == ROAD_COLOR, axis=2)
    sidewalk_mask = np.all(band == SIDEWALK_COLOR, axis=2)

    total_pixels = road_mask.size
    if total_pixels == 0:
        return False  

    non_road_ratio = np.sum(~road_mask) / total_pixels

    return non_road_ratio > 0.2


def oncoming_car_danger(label_img, ego_roi):
    h, w, _ = label_img.shape
    car_color = SEMANTIC_COLORS["car"]

    y1, y2, x1, x2 = oncoming_traffic_roi(h, w)
    roi = label_img[y1:y2, x1:x2]

    car_mask = np.all(roi == car_color, axis=2)
    car_ratio = np.sum(car_mask) / car_mask.size

    if car_ratio < ONCOMING_CAR_THRESHOLD:
        return False, car_ratio, (x1, y1, x2, y2)

    separated = separated_by_non_road(label_img, ego_roi, (x1, y1, x2, y2))

    danger = separated
    return danger, car_ratio, (x1, y1, x2, y2)

def detect_left_turn(label_img):
    """
    Detects left turn by checking the horizontal shift of the road in the lower half.
    Returns True if a left turn is detected.
    """
    h, w, _ = label_img.shape

    # Focus on lower half of the road
    y1, y2 = int(h * 0.6), int(h * 0.9)
    roi = label_img[y1:y2, :]

    road_mask = np.all(roi == ROAD_COLOR, axis=2)
    if np.sum(road_mask) < 500:  # road not visible
        return False

    centers = []
    for row in range(road_mask.shape[0]):
        xs = np.where(road_mask[row])[0]
        if len(xs) == 0:
            continue
        center_x = np.mean(xs)
        centers.append(center_x)

    if len(centers) < 5:
        return False

    # Check trend: if average center moves right -> left turn
    slope = centers[-1] - centers[0]
    threshhold = w * 0.14 
    return slope > threshhold 

# =========================================================
# Final brake decision (LABEL-ONLY)
# =========================================================
# =========================================================
# Final brake decision (EXTENDED)
# =========================================================
def compute_brake(label_img, prev_car_ratio):
    car_risk, car_ratio, ego_roi = car_danger(label_img, prev_car_ratio)
    ped_risk, ped_roi = pedestrian_danger(label_img)

    oncoming_risk, oncoming_ratio, oncoming_roi = oncoming_car_danger(label_img, ego_roi)
    left_turning = detect_left_turn(label_img)

    left_turn_danger = oncoming_risk
    left_turn_brake = oncoming_risk and left_turning

    brake1 = int(car_risk)
    brake2 = int(ped_risk)
    brake3 = int(left_turn_brake)
    return (
        brake1,
        brake2,
        brake3,
        car_ratio,
        ego_roi,
        ped_roi,
        oncoming_roi,
        left_turn_danger
    )

# =========================================================
# Video generator (logic from labels)
# =========================================================
def video_generator(video_folders):
    for video_folder in video_folders:
        frame_files = sorted(video_folder.glob("*.png"))
        prev_car_ratio = None

        for idx, frame_path in enumerate(frame_files):
            label_path = labels_root / video_folder.name / (frame_path.stem + ".png")
            if not label_path.exists():
                continue

            frame = cv2.imread(str(frame_path))
            label = cv2.imread(str(label_path))

            if frame is None or label is None:
                continue

            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
            label = cv2.resize(
                label,
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                interpolation=cv2.INTER_NEAREST
            )

            brake1, brake2, brake3, car_ratio, roi_coords, ped_roi, oncoming_roi, left_turn_danger = compute_brake(label, prev_car_ratio)
            prev_car_ratio = car_ratio

            yield video_folder.name, idx, frame, brake1, brake2, brake3, roi_coords, ped_roi, oncoming_roi, left_turn_danger

# =========================================================
# Visualization (DISPLAY ONLY)
# =========================================================
def visualize(video_list, display_size=(800, 600), pause_time=40):
    left_turn=False
    for video_folder in video_list:
        for vname, idx, frame, brake1, brake2, brake3, roi_coords, ped_roi, oncoming_roi, left_turn_danger in video_generator([video_folder]):

            frame_disp = cv2.resize(frame, display_size)

            sx = display_size[0] / frame.shape[1]
            sy = display_size[1] / frame.shape[0]

            x1, y1, x2, y2 = roi_coords
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if brake1:
                cv2.putText(
                    frame_disp, "green",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.8, (0, 0, 255), 4
                )

            px1, py1, px2, py2 = ped_roi
            px1, px2 = int(px1 * sx), int(px2 * sx)
            py1, py2 = int(py1 * sy), int(py2 * sy)
            cv2.rectangle(frame_disp, (px1, py1), (px2, py2), (255, 0, 255), 2)

            if brake2:
                cv2.putText(
                    frame_disp, "purple",
                    (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.8, (0, 0, 255), 4
                )

            ox1, oy1, ox2, oy2 = oncoming_roi
            ox1, ox2 = int(ox1 * sx), int(ox2 * sx)
            oy1, oy2 = int(oy1 * sy), int(oy2 * sy)
            cv2.rectangle(frame_disp, (ox1, oy1), (ox2, oy2), (255, 255, 0), 2)

            if left_turn == True:
                cv2.putText(
                    frame_disp, "LEFT TURN ON!",
                    (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 4
                )
                if left_turn_danger:
                    cv2.putText(
                        frame_disp, "LEFT TURN DANGER!",
                        (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 4
                    )

                if brake3:
                    cv2.putText(
                        frame_disp, "blue",
                        (540, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 4
                    )

            cv2.imshow(video_folder.name, frame_disp)

            key = cv2.waitKey(pause_time) & 0xFF

            if key == ord('p'):
                pause_time = 0 if pause_time > 0 else 40

            if key == ord('a'):
                left_turn = not left_turn

            elif key == ord('q'):
                cv2.destroyAllWindows()
                return

        cv2.destroyWindow(video_folder.name)

# =========================================================
# Run
# =========================================================
video_folders = sorted([f for f in frames_root.iterdir() if f.is_dir()])
print("Press 'q' to quit.")
visualize(video_folders[2:3])

####### =================================================
# de incorporat buton de semnalizare stanga dreapta in interfata
# si logica de detectie a masinilor care vin din sens opus la viraj
# daca butonul e apasat si e detectata masina care vine din sens opus
# sa se activeze left turn_danger si sa franeze robotul

