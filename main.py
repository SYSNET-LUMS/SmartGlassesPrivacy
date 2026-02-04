# Additional imports
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
import base64
import json
import numpy as np
import sys
import os
import uuid
import msgpack, zlib, json

# Helper: compute center of box
def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# Encryption utility functions
def pad(data, block_size=16):
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)

def encrypt_data_aes128(data, key):
    cipher = AES.new(key, AES.MODE_ECB)
    padded_data = pad(data, AES.block_size)
    encrypted = cipher.encrypt(padded_data)
    return encrypted

# Load public key from PEM file
with open("encryption/public_key.pem", "rb") as f:
    public_key = RSA.import_key(f.read())

# No need to save the key now
# Save the key only once if it doesn't exist
# if not os.path.exists(KEY_PATH):
#     with open(KEY_PATH, "wb") as key_file:
#         key_file.write(AES_KEY)
# else:
#     with open(KEY_PATH, "rb") as key_file:
#         AES_KEY = key_file.read()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
import src.blur_functions as bf
import src.utils as utils
import subprocess
import torch
from src.backbones import get_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Config:
    INPUT_VIDEO_PATH = "./input/test.mov"
    # INPUT_VIDEO_PATH = 0
    OUTPUT_VIDEO_PATH = "./output/blurred.mp4"
    SAVE_OUTPUT = True
    AUDIO = False
    DETECTOR_MODEL = "yunet" ## or "yunet"
    # EMBEDDING_MODEL = "ghost" # or "edgeface_s_gamma_05"
    EMBEDDING_MODEL = "edgeface_s_gamma_05"
    EMBEDDING_MODEL_PATH = "./src/models/edgeface_s_gamma_05.pt"
    # EMBEDDING_MODEL_PATH = "../src/models/GN_W0.5_S2_ArcFace_epoch16.h5"
    FRAME_SKIP = 5
    TRACKING_SKIP = 0
    BLUR_ENABLED = True
    ROTATION_DEGREES = 90
    

    LANDMARK_STALENESS_THRESHOLD = 15
    BOX_STALENESS_THRESHOLD = 15
    MSE_THRESHOLD = 150

    # SCRFD Config
    SCRFD_MODEL_PATH = "./src/models/det_500m.onnx"
    SCRFD_INPUT_SIZE = (640, 640)

    # --- Visualization & Logging ---
    OVERLAY_LANDMARKS = False
    OVERLAY_DETECTOR_BOX = False
    DISPLAY_VIDEO = True
    
    LANDMARKS_PATH = "./Synthetic Replacement/landmarks_root"

total_inference_times = []
total_blur_times = []
total_detector_times = []
total_landmarker_times = []
total_scale_times = []
total_tracking_times = []
total_mse_times = []
total_encryption_times = []
enc_times = []
total_frame_read_times = []
total_frame_write_times = []

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if __name__ == "__main__":
    # try:
    # All 3 Ml models be pre-loaded in the glasses at boot time
        config = Config()
        detector = utils.initialize_detector(config)
        interpretor, input_details, output_details = utils.initialize_landmark_detector()
        
        if config.EMBEDDING_MODEL == "edgeface_s_gamma_05":
            model_name = "edgeface_s_gamma_05"
            model = get_model(model_name)
            model.load_state_dict(torch.load(config.EMBEDDING_MODEL_PATH, map_location='cpu'))
            model.eval()
        # elif config.EMBEDDING_MODEL == "ghost":
            # model = GhostFaceNets.buildin_models("ghostnetv1", dropout=0, emb_shape=512, output_layer='GDC', bn_momentum=0.9, bn_epsilon=1e-5)
            # model = GhostFaceNets.add_l2_regularizer_2_model(model, weight_decay=5e-4, apply_to_batch_normal=False)
            # model = GhostFaceNets.replace_ReLU_with_PReLU(model, target_activation='PReLU')
            # model.load_weights(config.EMBEDDING_MODEL_PATH)
        
        cap, out, orig_frame_width, orig_frame_height = utils.initialize_video_io(config)

        start_process_time = time.perf_counter()
        frame_count = 0
        blur_count = 0
        detector_count = 0
        landmarker_count = 0
        tracking_count = 0
        mse_thresh_count = 0

        detected_faces_boxes = []
        landmark_boxes = []
        last_landmark_boxes = []

        frames_since_last_landmark = 0
        frames_since_last_box = 0

        prev_gray = None
        prev_frame = None

        face_metadata = {}  # sidecar dict to collect encrypted metadata
        landmarks_data = {}
        landmarks_data_normal = {}

        face_tracks = {}  # face_id: {"box": (x1,y1,x2,y2), "aes_key": key}
        MAX_CENTER_DIST = 60 # Vary this

        face_id_to_aes_key = {}

        while cap.isOpened():
            st = time.perf_counter()
            ret, frame = cap.read()
            en = time.perf_counter()
            total_frame_read_times.append((en-st)*1000)

            if not ret:
                break

            frame_start_time = time.perf_counter()
            frame_count += 1

            mse_values = []
            if prev_frame is not None and detected_faces_boxes:
                for face_region in detected_faces_boxes:
                    st = time.perf_counter()
                    mse = utils.calculate_frame_diff(prev_frame, frame, face_region)
                    en = time.perf_counter()
                    total_mse_times.append((en-st)*1000)
                    mse_values.append(mse)

            run_detector = (
                frame_count == 1 or 
                frame_count % (config.FRAME_SKIP + 1) == 0 or 
                (mse_values and max(mse_values) > config.MSE_THRESHOLD)
            )
            if (mse_values and max(mse_values) > config.MSE_THRESHOLD):
                mse_thresh_count += 1

            current_height, current_width = frame.shape[:2]
            rotated_frame = frame

            if config.INPUT_VIDEO_PATH != 0:
                if config.ROTATION_DEGREES == 90 and current_width > current_height:
                    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif config.ROTATION_DEGREES == 270 and current_width > current_height:
                    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            output_frame = rotated_frame
            output_h, output_w = output_frame.shape[:2]
            rgb_frame = output_frame
            prev_frame = output_frame

            if run_detector:
                detector_count += 1
                start_detector_time = time.perf_counter()
                faces = utils.detect_faces(rgb_frame, detector, config)
                end_detector_time = time.perf_counter()
                total_detector_times.append((end_detector_time - start_detector_time) * 1000)
                detected_faces_boxes = [utils.expand_box_with_margin(face["box"], margin=0.1, frame_width=output_w, frame_height=output_h) for face in faces]
                detected_face_scores = [face["score"] for face in faces]
                
                if detected_faces_boxes:
                    frames_since_last_box = 0
                else:
                    frames_since_last_box += 1
                    if frames_since_last_box >= config.BOX_STALENESS_THRESHOLD:
                        detected_faces_boxes = []
                    face_tracks = {}
            else:
                frames_since_last_box += 1
                if frames_since_last_box >= config.BOX_STALENESS_THRESHOLD:
                    detected_faces_boxes = []
                if prev_gray is not None:
                    if frame_count % (config.TRACKING_SKIP+1) == 0:
                        start = time.perf_counter()
                        detected_faces_boxes = [utils.track_box_with_optical_flow(prev_gray, rgb_frame, box, lk_params, scale_factor=0.4)[0] for box in detected_faces_boxes] 
                        end = time.perf_counter()
                        tracking_count += 1
                        total_tracking_times.append((end-start)*1000)

            landmark_boxes = []
            current_landmarks_found = False

            new_tracks = {}

            for i, box in enumerate(detected_faces_boxes):
                cx, cy = box_center(box)

                matched_id = None
                min_dist = float("inf")
                for face_id, info in face_tracks.items():
                    pcx, pcy = box_center(info["box"])
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array([pcx, pcy]))
                    if dist < MAX_CENTER_DIST and dist < min_dist:
                        matched_id = face_id
                        min_dist = dist

                if matched_id:
                    new_tracks[matched_id] = {"box": box, "aes_key": face_tracks[matched_id]["aes_key"]}
                else:
                    # Assign new face ID and AES key
                    new_id = str(uuid.uuid4())
                    new_key = get_random_bytes(16)
                    # print("\n\n\n", new_key, "\n\n\n")
                    face_id_to_aes_key[new_id] = new_key  # store plain key

                    new_tracks[new_id] = {
                        "box": box,
                        "aes_key": new_key,
                    }
                    

                x1, y1, x2, y2 = box
                if x1 < x2 and y1 < y2:
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    x2 = min(x2, current_width)
                    y2 = min(y2, current_height)
                    face_roi_rgb = rgb_frame[y1:y2, x1:x2]
                    if face_roi_rgb.size > 0:
                        start_landmarker_time = time.perf_counter()
                        landmarks, score = utils.get_landmarks_interpretor_with_score(face_roi_rgb, interpretor, input_details, output_details)
                        end_landmarker_time = time.perf_counter()
                        total_landmarker_times.append((end_landmarker_time - start_landmarker_time) * 1000)
                        landmarker_count += 1

                        frames_since_last_landmark = 0
                        current_landmarks_found = True
                        landmarks_scaled = utils.scale_landmarks(landmarks, x1, y1, x2 - x1, y2 - y1)
                        landmark_boxes.append(landmarks)
                        last_landmark_boxes = landmark_boxes.copy()

                        #---- encryption                        
                        # Save original face before blur for encryption
                        face_id = matched_id if matched_id else new_id
                        track_info = new_tracks[face_id]
                        aes_key = track_info["aes_key"]
                        # encrypted_box = track_info["box"]
                        # x1, y1, x2, y2 = encrypted_box

                        # new_x1, new_y1, new_x2, new_y2 = utils.expand_box_with_margin((x1, y1, x2, y2), margin=0.2, frame_width=output_w, frame_height=output_h)
                        
                        frame_height, frame_width = frame.shape[:2]
                        hull = cv2.convexHull(landmarks_scaled)
                        x, y, w, h = cv2.boundingRect(hull)
                        
                        # Ensure the roi is within frame boundaries
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame_width - x)
                        h = min(h, frame_height - y)

                        landmarks_adjusted = np.array(landmarks_scaled)  # shape: (468, 3)
                        landmarks_adjusted[:, 0] -= x  # x
                        landmarks_adjusted[:, 1] -= y  # y
                        
                        landmarks_normal = landmarks_adjusted / np.array([w, h])
                        
                        st_enc = time.perf_counter()
                        face_image_for_encryption = rgb_frame[y:y+h, x:x+w]

                        _, buffer = cv2.imencode('.jpg', face_image_for_encryption)
                        face_image_encoded = base64.b64encode(buffer).decode('utf-8')
                        data = {
                            "frame": frame_count,
                            # "landmarks": np.array(landmarks_adjusted).tolist(),
                            "face_image": face_image_encoded,
                            "box": [x, y, x+w, y+h],
                            "box_score": float(detected_face_scores[i]),
                            "land_score": float(score),
                            "face_id": face_id
                        }
                        data_land = {
                            "frame": frame_count,
                            "landmarks": np.array(landmarks_adjusted).tolist(),
                        }
                        
                        data_land_normal = {
                            "frame": frame_count,
                            "landmarks": landmarks_normal.tolist(),
                        }
                        
                        # json_data = json.dumps(data).encode('utf-8')
                        # encrypted = encrypt_data_aes128(json_data, aes_key)
                        
                        # Initialize frame entry if not already
                        if face_id not in face_metadata:
                            face_metadata[face_id] = []

                        if face_id not in landmarks_data:
                            landmarks_data[face_id] = []
                            landmarks_data_normal[face_id] = []

                        face_metadata[face_id].append(data)
                        landmarks_data[face_id].append(data_land)
                        landmarks_data_normal[face_id].append(data_land_normal)
                        en_enc = time.perf_counter()
                        enc_times.append((en_enc - st_enc) * 1000)
                        
                    if config.BLUR_ENABLED:
                        start_blur_time = time.perf_counter()
                        output_frame = bf.apply_blur_new(rgb_frame, hull, x, y, w, h)
                        # output_frame = bf.apply_gaussian_blur(rgb_frame, landmarks_scaled)
                        end_blur_time = time.perf_counter()
                        total_blur_times.append((end_blur_time - start_blur_time) * 1000)
                        blur_count += 1

            face_tracks = new_tracks

            if config.OVERLAY_DETECTOR_BOX:
                 for (x1, y1, x2, y2) in detected_faces_boxes:
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if config.OVERLAY_LANDMARKS:
                color = (0, 255, 0)
                for landmarks in landmark_boxes:
                    for lx, ly in landmarks:
                        cv2.circle(output_frame, (lx, ly), 1, color, -1)
            
            prev_gray = output_frame

            if config.DISPLAY_VIDEO:
                display_frame = cv2.resize(output_frame, (512, 512))
                cv2.imshow("Face Blurring Output", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_end_time = time.perf_counter()
            
            if config.SAVE_OUTPUT and out:
                st = time.perf_counter()
                out.write(output_frame)
                en = time.perf_counter()
                total_frame_write_times.append((en-st)*1000)
            
            inference_time_ms = (frame_end_time - frame_start_time) * 1000
            total_inference_times.append(inference_time_ms)

        # Encrypt all collected metadata at the end, could have been done during the loop as well.
        encrypted_face_metadata = {}
        to_be_embedded = {}
        
        st = time.perf_counter()
        for face_id, entries in face_metadata.items():
            aes_key = face_id_to_aes_key[face_id]
            best_entry = entries[0]
            best_land_score = best_entry.get("land_score", -1)
            best_box_score = best_entry.get("box_score", -1)
            max_index = 0
            # why encrypt each entry when i can just encrypt every time this guys face shows up at once.
            for i, entry in enumerate(entries):
                land_score = entry.get("land_score", -1)
                box_score = entry.get("box_score", -1)

                # Priority: land_score first, then box_score
                # if (land_score > best_land_score) or \
                # (land_score == best_land_score and box_score > best_box_score):
                if (land_score > best_land_score):
                    best_land_score = land_score
                    best_box_score = box_score
                    max_index = i
            to_be_embedded[face_id] = (entries[max_index].get("face_image", None), landmarks_data[face_id][max_index].get("landmarks", None))
            data = msgpack.packb(entries)
            encrypted = encrypt_data_aes128(data, aes_key)

            encrypted_face_metadata[f"frame_{face_id}"] = encrypted

        en = time.perf_counter()
        print(f"Encryption metadata processing took: {(en-st)*1000:.2f} ms")
        
        st = time.perf_counter()
        batch_tensor, face_id_order = utils.preprocess_face_images(to_be_embedded, config.EMBEDDING_MODEL)
        en = time.perf_counter()
        print("Preprocessing and allignment: ", (en-st)*1000, " ms")

        if config.EMBEDDING_MODEL == "edgeface_s_gamma_05":
            with torch.no_grad():
                st = time.perf_counter()
                embeddings = model(batch_tensor)  # shape: (12, emb_dim)
                en = time.perf_counter()
                print(f"Batched inference time for {len(face_id_order)} images: {(en - st) * 1000:.2f} ms")
        else:
            # Run inference
            st = time.perf_counter()
            embeddings = model.predict(batch_tensor)
            en = time.perf_counter()
            print(f"Batched inference time for {len(face_id_order)} images: {(en - st) * 1000:.2f} ms")
    # print("\n\n\n",output_tensor.shape,"\n\n\n")
        
        if config.EMBEDDING_MODEL == "edgeface_s_gamma_05":
            face_embeddings = embeddings.numpy().tolist()  # Shape: (N, 512)
        else:
            face_embeddings = embeddings.tolist()
            
        # New dicts to store AES key + embedding
        face_id_to_combined_key_embedding = {}
        cipher_rsa = PKCS1_OAEP.new(public_key)

        for i, face_id in enumerate(face_id_order):
            aes_key = face_id_to_aes_key[face_id]   # This is the raw AES key (bytes)
            embedding = face_embeddings[i]          # This is a list of 512 floats

            # Encrypt the AES key with RSA
            encrypted_aes_key = cipher_rsa.encrypt(aes_key)
            encrypted_aes_key_b64 = base64.b64encode(encrypted_aes_key).decode('utf-8')

            # Encrypt the embedding with AES
            encrypted_embedding_b64 = encrypt_data_aes128(json.dumps(embedding).encode('utf-8'), aes_key)

            # Store raw AES key + embedding
            face_id_to_combined_key_embedding[face_id] = {
                "aes_key": encrypted_aes_key_b64,  # Encode for JSON safety
                "embedding": encrypted_embedding_b64
            }
        
        video_identifier = str(uuid.uuid4())
        face_id_to_combined_key_embedding["video_identifier"] = video_identifier

        # Save if needed
        st = time.perf_counter()

        if config.SAVE_OUTPUT:
            # Landmark storage for Tier 2 testing (development)
            utils.convert_json_to_structure(config.LANDMARKS_PATH, landmarks_data_normal, config.INPUT_VIDEO_PATH)
        
        # Landmarks
        with open("./output/landmarks.msgpack", "wb") as f:
            f.write(zlib.compress(msgpack.packb(landmarks_data)))

        # Face keys
        with open("./output/face_keys.msgpack", "wb") as f:
            f.write(zlib.compress(msgpack.packb(face_id_to_combined_key_embedding)))

        # Encrypted metadata
        with open("./output/video_metadata_encrypted.msgpack", "wb") as f:
            f.write(zlib.compress(msgpack.packb(encrypted_face_metadata)))

        end = time.perf_counter()
        tme = (end - st) * 1000
        end_process_time = time.perf_counter()
    
        print(f"Saving files took: {tme:.2f} ms")
        
        if config.AUDIO:
            print("Merging audio...")
            temp_audio_path = "./output/temp_audio.aac"
            temp_output_path = "./output/temp_blurred_with_audio.mp4"  # Temporary output
            
            # Extract audio from original video
            print(f"Extracting audio from {config.INPUT_VIDEO_PATH}...")
            extract_audio_cmd = [
                "ffmpeg", "-y", "-i", config.INPUT_VIDEO_PATH, "-vn",
                "-acodec", "copy", temp_audio_path
            ]
            subprocess.run(extract_audio_cmd, check=True)

            # Merge audio with processed video to temporary file
            utils.merge_audio_with_video(
                config.OUTPUT_VIDEO_PATH,  # Input video
                temp_audio_path,           # Audio source  
                temp_output_path,          # Temporary output file
                "1021k",                   # Video bitrate
                "101k"                     # Audio bitrate
            )
            
            # Replace original with the new version
            import os
            os.replace(temp_output_path, config.OUTPUT_VIDEO_PATH)
            
            # Clean up temporary files
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
            print(f"Final video with audio saved to: {config.OUTPUT_VIDEO_PATH}")
        
        # elif config.SAVE_OUTPUT:
        #     print("Compressing to match original bitrate...")
        #     import subprocess
        #     import os
            
        #     compressed_path = config.OUTPUT_VIDEO_PATH.replace('.mp4', '_compressed.mp4')
            
        #     cmd = [
        #         'ffmpeg', '-y', '-i', config.OUTPUT_VIDEO_PATH,
        #         '-c:v', 'libx264',
        #         '-b:v', '1021k',      # Match your original video bitrate
        #         '-maxrate', '1021k',   # Prevent bitrate spikes
        #         '-bufsize', '2042k',   # Buffer size (2x bitrate)
        #         '-c:a', 'aac',
        #         '-b:a', '101k',       # Audio bitrate (1122 - 1021 = 101k)
        #         compressed_path
        #     ]
            
        #     try:
        #         subprocess.run(cmd, check=True, capture_output=True)
        #         os.replace(compressed_path, config.OUTPUT_VIDEO_PATH)
        #         print("Bitrate matched successfully!")
        #     except Exception as e:
        #         print(f"Compression failed: {e}")
        
        if total_inference_times:
            avg_inference = sum(total_inference_times) / len(total_inference_times)
            avg_fps = 1000 / avg_inference if avg_inference > 0 else 0
            print(f"--- Processing Summary ---")
            print(f"Total frames processed: {frame_count}")
            print(f"Frames processed by Detector: {detector_count}")
            print(f"Frames Tracked: {tracking_count}")
            print(f"Frames processed by landmarker: {landmarker_count}")
            print(f"Frames hit by mse thresh: {mse_thresh_count}")
            print(f"Frames Blurred: {blur_count}")
            print(f"Detector Duration: {(sum(total_detector_times) / len(total_detector_times)):.4f} ms")
            print(f"Landmarker Duration: {(sum(total_landmarker_times) / len(total_landmarker_times)):.4f} ms")
            print(f"Blur Duration: {(sum(total_blur_times) / len(total_blur_times)):.4f} ms")
            print(f"Tracking Duration: {(sum(total_tracking_times)/len(total_tracking_times)):.4f} ms")
            print(f"MSE Duration: {(sum(total_mse_times)/len(total_mse_times)):.4f} ms")
            print(f"Average enc time: {sum(enc_times) / len(enc_times):.4f} ms" if enc_times else "Average enc time: N/A")
            print(f"Average Frame Read Time: {(sum(total_frame_read_times)/len(total_frame_read_times)):.4f} ms")
            print(f"Average Frame Write Time: {(sum(total_frame_write_times)/len(total_frame_write_times)):.4f} ms")
            if total_encryption_times:
                print(f"Encryption Duration: {(sum(total_encryption_times)/len(total_encryption_times)):.4f} ms")
            print(f"Average Frame Processing Time: {avg_inference:.2f} ms")
            print(f"Average FPS for processing loop: {avg_fps:.2f}")
            print(f"Total processing duration: {end_process_time - start_process_time:.2f} seconds")
            print(f"Effective FPS: {frame_count / (end_process_time - start_process_time):.2f}")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     sys.exit(1)
    # finally:
    #     if 'cap' in locals() and cap.isOpened():
    #         cap.release()
    #     if 'out' in locals() and out is not None:
    #         out.release()
    #     cv2.destroyAllWindows()
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        cv2.destroyAllWindows()  