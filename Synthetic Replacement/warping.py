import os
import cv2
import numpy as np
import json
import time
from tqdm import tqdm
import shutil

class FaceSwapper:
    def __init__(self, synth_img_folder, synth_lm_folder):
        self.synth_faces = []
        self.synth_landmarks = []
        self.logged_match = False
        self.cached_triangles = None
        self.cached_w, self.cached_h = None, None
        
        for fname in sorted(os.listdir(synth_img_folder)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(synth_img_folder, fname)
                lm_path = os.path.join(synth_lm_folder, os.path.splitext(fname)[0] + ".txt")
                if os.path.exists(lm_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        self.synth_faces.append((fname, img))
                        self.synth_landmarks.append(lm_path)

        if not self.synth_faces:
            raise ValueError("⚠️ No synthetic face + landmark pairs found!")

        print(f"Loaded {len(self.synth_faces)} synthetic faces")

    @staticmethod
    def load_landmarks(file_path, img_shape):
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            pts = np.array([[lm['x'], lm['y']] for lm in data], dtype=np.float32)
            if pts.max() <= 1.0:
                h, w = img_shape[0], img_shape[1]
                pts[:, 0] *= w
                pts[:, 1] *= h
        else:
            pts = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
            if pts.ndim == 1:
                pts = pts.reshape(-1, 2)
            if pts.max() <= 1.0:
                h, w = img_shape[0], img_shape[1]
                pts[:, 0] *= w
                pts[:, 1] *= h
        return pts

    @staticmethod
    def mean_face_color(img, lm_path):
        """Compute mean color inside face convex hull."""
        pts = FaceSwapper.load_landmarks(lm_path, img.shape)[:468]
        mask = np.zeros(img.shape[:2], np.uint8)
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        mean_color = cv2.mean(img, mask=mask)[:3]  # BGR
        return mean_color

    def best_match_synthetic(self, frame, target_bgr):
        """Find synthetic face whose mean color is closest to a given BGR color."""
        frame_mean = np.array(target_bgr, dtype=np.float32)

        best_idx, best_dist = -1, float("inf")
        scores = []

        for i, (synth_name, synth_img) in enumerate(self.synth_faces):
            synth_lm = self.synth_landmarks[i]
            synth_mean = self.mean_face_color(synth_img, synth_lm)
            dist = np.linalg.norm(frame_mean - np.array(synth_mean))
            scores.append((synth_name, dist))

            if dist < best_dist:
                best_idx, best_dist = i, dist

        if not self.logged_match:
            print("🎨 Color distances:")
            for name, d in scores:
                print(f"   {name}: {d:.2f}")
            print(f"✅ Best match: {self.synth_faces[best_idx][0]}, distance {best_dist:.2f}")
            self.logged_match = True
        
        save_dir = os.path.join(os.getcwd(), "best_match")
        os.makedirs(save_dir, exist_ok=True)

        chosen_name, chosen_img = self.synth_faces[best_idx]
        chosen_lm_path = self.synth_landmarks[best_idx]
        ext = os.path.splitext(chosen_name)[1] or ".jpg"
        img_out = os.path.join(save_dir, f"best_{chosen_name}")
        try:
            ok = cv2.imwrite(img_out, chosen_img)
            if ok:
                print(f"💾 Saved best-match image: {img_out}")
            else:
                print("⚠️ Failed to write best-match image with cv2.imwrite")
        except Exception as e:
            print(f"⚠️ Could not save best-match image: {e}")

        try:
            lm_ext = os.path.splitext(chosen_lm_path)[1] or ".txt"
            lm_out = os.path.join(
                save_dir, f"best_{os.path.splitext(chosen_name)[0]}_landmarks{lm_ext}"
            )
            shutil.copyfile(chosen_lm_path, lm_out)
            print(f"💾 Saved best-match landmarks: {lm_out}")
        except Exception as e:
            print(f"⚠️ Could not save best-match landmarks: {e}")
        return self.synth_faces[best_idx], self.synth_landmarks[best_idx]

    def process_video(self, video_path, landmarks_root, output_folder,target_bgr):
        self.cached_triangles = None
        self.cached_w, self.cached_h = None, None
        self.logged_match = False

        total_start = time.time() 
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_folder, f"{video_name}.mp4")
        os.makedirs(output_folder, exist_ok=True)

        landmarks_video_folder = os.path.join(landmarks_root, video_name)
        if not os.path.exists(landmarks_video_folder):
            print(f"⚠️ No landmarks folder found for video {video_name}.mp4")
            return

        setup_start = time.time()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        setup_end = time.time()
        print(f"⏱ Setup time: {setup_end - setup_start:.3f} sec")

        ret, first_frame = cap.read()
        if not ret:
            return
        best_start = time.time()
        (synth_name, synth_img), synth_lm_path = self.best_match_synthetic(first_frame,target_bgr)
        synth_points = self.load_landmarks(synth_lm_path, synth_img.shape)[:468]
        best_end = time.time()
        print(f"⏱ Synthetic match time: {best_end - best_start:.3f} sec")

        # Reset capture to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_times = []
        timings_accum = {"landmark_load": [], "warp": [], "seamless_clone": [], "total": []}

        for frame_num in tqdm(range(total_frames), desc=f"Processing {video_name}"):
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()
            frame_name = f"frame_{frame_num}"
            canvas = frame.copy()

            face_folders = [f for f in os.listdir(landmarks_video_folder) 
                        if f.startswith('face_') and os.path.isdir(os.path.join(landmarks_video_folder, f))]

            for face_folder in sorted(face_folders):
                lm_path = os.path.join(landmarks_video_folder, face_folder, f"{frame_name}.txt")
                if os.path.exists(lm_path):
                    try:
                        working_img = canvas.copy()
                        result, timings = self.swap_face(working_img, lm_path, synth_img, synth_points)
                        canvas = result

                        for k in timings:
                            timings_accum[k].append(timings[k])

                        if frame_num == 0:
                            print("⏱ Timing breakdown (first frame):", {k: f"{v:.3f}" for k, v in timings.items()})

                    except Exception as e:
                        print(f"⚠️ Error processing {frame_name} in {face_folder}: {str(e)}")

            out.write(canvas)
            frame_end = time.time()
            frame_times.append(frame_end - frame_start)

        cap.release()
        out.release()

        total_elapsed = time.time() - total_start
        avg_frame_time = np.mean(frame_times) if frame_times else 0

        if timings_accum["total"]:
            avg_breakdown = {k: np.mean(v) for k, v in timings_accum.items()}
            print("⏱ Avg timing breakdown:", {k: f"{v:.3f}" for k, v in avg_breakdown.items()})

        print(f"✅ Processed {video_name} in {total_elapsed:.2f} sec "
              f"(avg {avg_frame_time:.3f} sec/frame, {1/avg_frame_time:.1f} FPS if >0)")

    def swap_face(self, dst_img, dst_lm_path, src_img, src_points):
        timings = {}
        t0 = time.time()

        # 1. Load landmarks
        t1 = time.time()
        dst_points = self.load_landmarks(dst_lm_path, dst_img.shape)[:468]
        timings["landmark_load"] = time.time() - t1

        h, w = dst_img.shape[:2]

        # 2. Alignment + warping
        t2 = time.time()
        KEY = [33, 263, 1]
        A, B = src_points[KEY], dst_points[KEY]
        M, _ = cv2.estimateAffinePartial2D(A, B, method=cv2.LMEDS)

        if M is not None:
            aligned_src = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            aligned_pts = cv2.transform(src_points[None, :, :], M)[0]
        else:
            aligned_src = src_img.copy()
            aligned_pts = src_points.copy()

        # ✅ cache triangulation once per video
        if self.cached_triangles is None or self.cached_w != w or self.cached_h != h:
            subdiv = cv2.Subdiv2D((0, 0, w, h))
            for p in dst_points:
                subdiv.insert((float(p[0]), float(p[1])))
            triangles_raw = subdiv.getTriangleList().reshape(-1, 3, 2)

            triangles_idx = []
            for tri in triangles_raw:
                idx = [int(np.argmin(np.linalg.norm(dst_points - pt, axis=1))) for pt in tri]
                if len(set(idx)) == 3:
                    triangles_idx.append(idx)

            self.cached_triangles = triangles_idx
            self.cached_w, self.cached_h = w, h

        canvas = dst_img.copy()
        for tri in self.cached_triangles:
            self.warp_triangle(aligned_src, canvas,
                              [aligned_pts[i] for i in tri],
                              [dst_points[i] for i in tri])
        timings["warp"] = time.time() - t2

        # 3. Seamless clone
        t3 = time.time()
        hull = cv2.convexHull(dst_points.astype(np.int32))
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        x, y, ww, hh = cv2.boundingRect(mask)
        center = (x + ww//2, y + hh//2)
        output = cv2.seamlessClone(canvas, dst_img, mask, center, cv2.MIXED_CLONE)
        timings["seamless_clone"] = time.time() - t3

        timings["total"] = time.time() - t0
        return output, timings


    def warp_triangle(self, img_src, img_dst, t_src, t_dst):
        r_src = self.safe_bounding_rect(t_src, img_src.shape)
        r_dst = self.safe_bounding_rect(t_dst, img_dst.shape)
        if r_src[2] <= 0 or r_src[3] <= 0 or r_dst[2] <= 0 or r_dst[3] <= 0:
            return

        t_src_offset = [(t_src[i][0]-r_src[0], t_src[i][1]-r_src[1]) for i in range(3)]
        t_dst_offset = [(t_dst[i][0]-r_dst[0], t_dst[i][1]-r_dst[1]) for i in range(3)]
        src_patch = img_src[r_src[1]:r_src[1]+r_src[3], r_src[0]:r_src[0]+r_src[2]]
        
        if src_patch.size == 0:
            return
            
        warp_mat = cv2.getAffineTransform(np.float32(t_src_offset), np.float32(t_dst_offset))
        warped_patch = cv2.warpAffine(src_patch, warp_mat, (r_dst[2], r_dst[3]), 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_dst_offset), (1.0, 1.0, 1.0), 16, 0)
        
        dst_y1, dst_y2 = r_dst[1], r_dst[1] + r_dst[3]
        dst_x1, dst_x2 = r_dst[0], r_dst[0] + r_dst[2]
        dst_y1 = max(0, dst_y1)
        dst_y2 = min(img_dst.shape[0], dst_y2)
        dst_x1 = max(0, dst_x1)
        dst_x2 = min(img_dst.shape[1], dst_x2)
        
        if dst_y2 <= dst_y1 or dst_x2 <= dst_x1:
            return
            
        dst_roi = img_dst[dst_y1:dst_y2, dst_x1:dst_x2]
        mask = cv2.resize(mask, (dst_x2-dst_x1, dst_y2-dst_y1))
        warped_patch = cv2.resize(warped_patch, (dst_x2-dst_x1, dst_y2-dst_y1))
        
        if dst_roi.shape[:2] == warped_patch.shape[:2] == mask.shape[:2]:
            img_dst[dst_y1:dst_y2, dst_x1:dst_x2] = (dst_roi * (1 - mask) + warped_patch * mask).astype(np.uint8)

    @staticmethod
    def safe_bounding_rect(points, img_shape):
        h, w = img_shape[:2]
        x, y, w_rect, h_rect = cv2.boundingRect(np.float32(points))
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        w_rect = max(1, min(w_rect, w-x))
        h_rect = max(1, min(h_rect, h-y))
        return (x, y, w_rect, h_rect)


def blurred_process_videos(videos_folder, landmarks_root, output_root, synth_img_folder, synth_lm_folder):
    swapper = FaceSwapper(synth_img_folder, synth_lm_folder)

    video_files = [f for f in os.listdir(videos_folder) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print("⚠️ No video files found in the input folder")
        return
    
    print(f"Found {len(video_files)} videos to process")
    video_path = os.path.join(videos_folder, video_files)
    swapper.process_video(video_path, landmarks_root, output_root)

