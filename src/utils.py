from src.scrfd import SCRFD
import subprocess
import sys
import tensorflow as tf
import cv2
import numpy as np
import os
import base64
from PIL import Image
from torchvision import transforms
import torch


# REFERENCE_FIVE_POINTS = np.array([
#     [38.2946, 51.6963],  # left eye
#     [73.5318, 51.5014],  # right eye
#     [56.0252, 71.7366],  # nose tip
#     [41.5493, 92.3655],  # mouth left
#     [70.7299, 92.2041],  # mouth right
# ], dtype=np.float32)

REFERENCE_FIVE_POINTS = np.array([
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
], dtype=np.float32)

def convert_json_to_structure(root_dir, data, video_path):
    for face_id, frames in data.items():
        # 2. Create the face directory: face_{id}
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        face_folder = os.path.join(root_dir, video_name, f"face_{face_id}")
        os.makedirs(face_folder, exist_ok=True)
        
        for entry in frames:
            frame_num = entry.get("frame")
            landmarks = entry.get("landmarks", [])
            
            # 3. Define the file path: frame_{num}.txt
            file_path = os.path.join(face_folder, f"frame_{frame_num}.txt")
            
            # 4. Write the landmarks in "x,y" format
            with open(file_path, "w") as f:
                for point in landmarks:
                    # Each point is a list like [x, y]
                    # if len(point) == 2:
                    f.write(f"{point[0]},{point[1]}\n")

def merge_audio_with_video(video_path, audio_source_path, output_path, video_bitrate="1021k", audio_bitrate="101k"):
    command = [
        "ffmpeg",
        "-y",  # Overwrite output if it exists
        "-i", video_path,        # Video input
        "-i", audio_source_path, # Audio input
        "-c:v", "libx264",       # Video codec
        "-b:v", video_bitrate,   # Video bitrate
        "-maxrate", video_bitrate,  # Max video bitrate
        "-bufsize", f"{int(video_bitrate.replace('k', '')) * 2}k",  # Buffer size
        "-c:a", "aac",           # Audio codec
        "-b:a", audio_bitrate,   # Audio bitrate
        "-map", "0:v:0",         # Video from first input
        "-map", "1:a:0",         # Audio from second input
        "-shortest",             # Stop when shortest stream ends
        output_path
    ]
    subprocess.run(command, check=True)

def initialize_detector(config):
    """Initializes the selected face detector."""
    detector = None
    if config.DETECTOR_MODEL == "scrfd":
        try:
            detector = SCRFD(model_file=config.SCRFD_MODEL_PATH) 
            detector.prepare(ctx_id=-1, input_size=config.SCRFD_INPUT_SIZE)
            print("SCRFD model loaded successfully.")
        except Exception as e:
            print(f"Error loading SCRFD model: {e}")
            sys.exit(1)
    elif config.DETECTOR_MODEL == "yunet":
        current_dir = os.path.dirname(os.path.abspath(__file__))  # directory of utils.py
        yunet_model_path = os.path.join(current_dir, "models", "face_detection_yunet_2023mar.onnx")
        
        detector = cv2.FaceDetectorYN.create(
            model=yunet_model_path, 
            config="", 
            input_size=(640, 640), 
            score_threshold=0.5,
            top_k=5000,
            nms_threshold=0.45
        )
    else:
        print(f"Unsupported detector model: {config.DETECTOR_MODEL}")
        sys.exit(1)
    return detector

def initialize_landmark_detector():
    """Initializes the MediaPipe Face Mesh landmark detector."""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # directory of utils.py
    landmarker_path = os.path.join(current_dir, "models", "MediaPipeFaceLandmarkDetector.tflite")

    interpreter = tf.lite.Interpreter(model_path=landmarker_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("MediaPipe Face Mesh initialized.")
    return interpreter, input_details, output_details

def initialize_video_io(config):
    """Initializes video capture and writer."""
    cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Input video properties: {frame_width}x{frame_height} @ {fps} FPS")

    output_width, output_height = frame_width, frame_height
    if config.ROTATION_DEGREES == 90 and frame_width > frame_height:
        output_width, output_height = frame_height, frame_width
        print(f"Video will be rotated {config.ROTATION_DEGREES} degrees clockwise.")
    elif config.ROTATION_DEGREES == 270 and frame_width > frame_height:
        output_width, output_height = frame_height, frame_width
        print(f"Video will be rotated {config.ROTATION_DEGREES} degrees counter-clockwise.")

    out = None
    if config.SAVE_OUTPUT:
        orig_bitrate = cap.get(cv2.CAP_PROP_BITRATE)
        bitrate_str = f"b;{orig_bitrate // 1000}k"
        print("Original Bitrate:", bitrate_str)
        # os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"] = f"vcodec;libx264|preset;medium|{bitrate_str}"

        # os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"] = "vcodec;libx264|crf;23|preset;medium|b;2000k"
        # fourcc = cv2.VideoWriter_fourcc(*codec_str)

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, (output_width, output_height))
        
        if not out.isOpened():
            print(f"Warning: Could not open video writer")
    
    return cap, out, frame_width, frame_height

def detect_faces(frame_rgb, detector, config):
    """Detects faces using the chosen detector and returns bounding boxes."""
    faces_data = []
    h, w = frame_rgb.shape[:2]

    if config.DETECTOR_MODEL == "scrfd":
        # print(f"Detector shape: {frame_rgb.shape}" )
        bboxes, scrfd_keypoints = detector.detect(frame_rgb,input_size=config.SCRFD_INPUT_SIZE)
        if bboxes is not None and len(bboxes) > 0:
            for i in range(bboxes.shape[0]):
                box = bboxes[i, :4].astype(int)
                score = bboxes[i, 4]
                key_points = scrfd_keypoints[i, :].reshape(-1, 2)
                # Add score if you want to filter later
                faces_data.append({"box": box, "score": score, "keypoints": key_points})
    elif config.DETECTOR_MODEL == "yunet":
        yunet_input = cv2.resize(frame_rgb, (640, 640))
        faces = detector.detect(yunet_input)
        if faces[1] is not None:
            for face in faces[1]:
                x, y, w_box, h_box = map(int, face[:4])
                score = float(face[-1]) 

                # Scale bounding box back to original image size
                x1 = int(x * (w / 640))
                y1 = int(y * (h / 640))
                x2 = int((x + w_box) * (w / 640))
                y2 = int((y + h_box) * (h / 640))

                keypoints = []
                for i in range(5):
                    kp_x = int(face[5 + i * 2] * (w / 640))
                    kp_y = int(face[5 + i * 2 + 1] * (h / 640))
                    keypoints.append([kp_x, kp_y])

                box = [x1, y1, x2, y2]
                faces_data.append({"box": box, "score": score, "keypoints": keypoints})
    else:
        print(f"Unsupported detector model: {config.DETECTOR_MODEL}")
    return faces_data

def get_landmarks(face_roi_rgb, mp_face_mesh):
    """Gets landmarks for a specific face ROI."""
    # print(face_roi_rgb.shape)
    face_mesh_result = mp_face_mesh.process(face_roi_rgb)
    if face_mesh_result.multi_face_landmarks:
        # Assuming only one face is processed by FaceMesh per ROI
        return face_mesh_result.multi_face_landmarks[0].landmark
    return None

def get_landmarks_interpretor(face_roi_rgb, interpreter, input_details, output_details):
    """Gets landmarks for a specific face ROI."""
    # print(face_roi_rgb.shape)
    face_crop = cv2.resize(face_roi_rgb, (192, 192))  # Adjust size based on model
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    input_tensor = face_rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]
    input_tensor = np.expand_dims(input_tensor, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[1]['index'])  # Shape: [1, 1404]
    landmarks = output_data[0]  # shape: (468, 3)
    return landmarks

def get_landmarks_interpretor_with_score(face_roi_rgb, interpreter, input_details, output_details):
    """Gets landmarks for a specific face ROI."""
    # print(face_roi_rgb.shape)
    face_crop = cv2.resize(face_roi_rgb, (192, 192))  # Adjust size based on model
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    input_tensor = face_rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]
    input_tensor = np.expand_dims(input_tensor, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[1]['index'])  # Shape: [1, 1404]
    landmarks = output_data[0]  # shape: (468, 3)
    score = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (1,) → float
    return landmarks, score

def expand_box_with_margin(box, margin, frame_width, frame_height):
    if margin == 0:
        return box
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1

    if isinstance(margin, float):  # margin is a percentage
        margin_x = int(box_w * margin)
        margin_y = int(box_h * margin)
    else:  # margin is pixels
        margin_x = margin_y = margin

    new_x1 = max(0, x1 - margin_x)
    new_y1 = max(0, y1 - margin_y)
    new_x2 = min(frame_width, x2 + margin_x)
    new_y2 = min(frame_height, y2 + margin_y)

    return new_x1, new_y1, new_x2, new_y2

def scale_landmarks(landmarks_norm, roi_x, roi_y, roi_w, roi_h, scale_x=1.0, scale_y=1.0):
    coords = landmarks_norm[:, :2]  # Only x and y, skip z
    coords[:, 0] = coords[:, 0] * roi_w * scale_x + roi_x
    coords[:, 1] = coords[:, 1] * roi_h * scale_y + roi_y
    return coords.astype(np.int32, copy=False)

def track_box_with_optical_flow(prev_frame, curr_frame, box, lk_params, scale_factor = 0.5):
    curr_frame = cv2.resize(curr_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    prev_frame = cv2.resize(prev_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
    
    step = 10
    pts = [[j, i] for i in range(y1, y2, step) for j in range(x1, x2, step)]
    if len(pts) == 0:
        return box, None, None

    box_pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    # Perform optical flow only on points inside the box
    new_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, box_pts, None, **lk_params)

    if status is None or np.sum(status) < 3:
        return box, None, None

    good_new = new_pts[status == 1]
    good_old = box_pts[status == 1]
    movement = good_new - good_old

    dx = np.median(movement[:, 0])
    dy = np.median(movement[:, 1])

    new_box = [int(x1 / scale_factor + dx / scale_factor), int(y1 / scale_factor + dy / scale_factor),
               int(x2 / scale_factor + dx / scale_factor), int(y2 / scale_factor + dy / scale_factor)]
    
    return new_box, good_new, good_old

def calculate_frame_diff(prev_frame, curr_frame, face_region):
    """Compute the Mean Squared Error (MSE) between two frames in the detected face region."""
    # Get frame dimensions
    frame_height, frame_width = prev_frame.shape[:2]
    
    # Extract and validate face region coordinates
    x1, y1, x2, y2 = face_region
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), frame_width)
    y2 = min(int(y2), frame_height)
    
    # Ensure the region is valid
    if x1 >= x2 or y1 >= y2:
        return float('inf')  # Return infinity for invalid regions
    
    try:
        # Crop the face regions
        prev_face = prev_frame[y1:y2, x1:x2]
        curr_face = curr_frame[y1:y2, x1:x2]
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_face, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_face, cv2.COLOR_BGR2GRAY)
        
        # Compute MSE
        mse = np.sum((prev_gray - curr_gray) ** 2) / float(prev_gray.shape[0] * prev_gray.shape[1])
        return mse
    except Exception as e:
        print(f"Error calculating frame difference: {e}")
        return float('inf')
    
def calculate_avg(times_list):
    return sum(times_list) / len(times_list) if times_list else 0

def align_face(img, landmarks_2d, target_size=(112, 112)):
    """
    Aligns the face using similarity transform based on 5 MediaPipe landmarks.
    - landmarks_2d: list of 468 landmarks with [x, y, z]
    """
    # Extract the 2D positions of the 5 key points
    try:
        src_pts = np.array([
            landmarks_2d[33][:2],   # left eye
            landmarks_2d[263][:2],  # right eye
            landmarks_2d[1][:2],    # nose tip
            landmarks_2d[61][:2],   # mouth left
            landmarks_2d[291][:2],  # mouth right
        ], dtype=np.float32)
    except IndexError:
        raise ValueError("Invalid landmark format or missing key points")

    # Compute the similarity transform
    dst_pts = REFERENCE_FIVE_POINTS
    M = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)[0]

    if M is None:
        raise ValueError("Affine transform could not be computed")

    # Warp the image
    aligned_img = cv2.warpAffine(img, M, target_size, borderValue=0.0)

    return aligned_img

def preprocess_face_images(to_be_embedded, embedding_model, target_size=(112, 112)):
    """
    Convert base64-encoded face images and landmarks into a batch tensor.
    Aligns the faces based on landmarks and overlays landmarks for visualization.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    preprocessed_images = []
    face_ids = []

    for face_id, (img_b64, landmarks_2d) in to_be_embedded.items():
        if img_b64 is None or landmarks_2d is None:
            continue

        # Decode base64
        img_bytes = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR

        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw landmarks for visualization
        # for entry in landmarks_2d:
        #     cv2.circle(img, (int(entry[0]), int(entry[1])), 1, (0, 255, 0), -1)

        # # Show image with landmarks
        # Image.fromarray(img).show()

        try:
            # Align the face
            aligned = align_face(img, landmarks_2d, target_size)
        except Exception as e:
            print(f"Alignment failed for {face_id}: {e}")
            continue

            
        # Convert to PIL and apply transform
        # Inside your preprocess_face_images loop
        # Image.fromarray(aligned).show()  # To confirm alignment visually
        if embedding_model == "edgeface_s_gamma_05":
            pil_img = Image.fromarray(aligned)
            tensor_img = transform(pil_img)
        else:
            pil_img = Image.fromarray(aligned)
            img_array = np.asarray(pil_img).astype('float32') / 255.0
            img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
            tensor_img = np.expand_dims(img_array, axis=0)  # Shape: (1, 112, 112, 3)


        preprocessed_images.append(tensor_img)
        face_ids.append(face_id)

    if not preprocessed_images:
        return None, []

    if embedding_model == "edgeface_s_gamma_05":
        batch_tensor = torch.stack(preprocessed_images)
    else:
        batch_tensor = np.concatenate(preprocessed_images, axis=0)
    return batch_tensor, face_ids
