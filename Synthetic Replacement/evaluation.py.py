import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy import linalg
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import lpips
import mediapipe as mp
from pathlib import Path
import pandas as pd
import warnings
import os

# --- CONFIGURATION SECTION ---
# Change these paths to point to your local data
CONFIG = {
    "PATHS": {
        "REFERENCE": "./data/original",     # Original ground truth videos
        "METHOD_A": "./data/method_mfs",    # Baseline results (e.g., MFS)
        "METHOD_B": "./data/method_ours",   # Your smart glasses project results
        "OUTPUT": "./eval_results"          # Where to save CSVs and summaries
    },
    "SETTINGS": {
        "USE_GPU": True,
        "MAX_FRAMES": 100,                  # Samples per video
        "TARGET_FPS": 5,                    # Sampling rate
        "CUDA_DEVICE": "0"
    }
}
# -----------------------------

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if CONFIG["SETTINGS"]["USE_GPU"]:
    os.environ['CUDA_VISIBLE_DEVICES'] = CONFIG["SETTINGS"]["CUDA_DEVICE"]

# MediaPipe eye landmark indices
LEFT_EYE_IDX = list(range(33, 133))
RIGHT_EYE_IDX = list(range(362, 462))
EYE_IDX = LEFT_EYE_IDX + RIGHT_EYE_IDX

class VideoQualityEvaluator:
    def __init__(self, use_gpu=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"[*] Initializing Evaluator on: {self.device}")
        
        # Load Models
        self.inception_model = self._load_inception_model()
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )

    def _load_inception_model(self):
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = nn.Identity()
        model.eval()
        return model.to(self.device)

    def extract_frames(self, video_path, target_fps=5):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0: return np.array([])
        
        frame_interval = max(1, int(round(original_fps / target_fps)))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1
        cap.release()
        return np.array(frames)

    def detect_and_crop_face(self, frame, padding=50):
        h, w = frame.shape[:2]
        results = self.face_detection.process(frame)
        if not results.detections: return None
        
        bbox = results.detections[0].location_data.relative_bounding_box
        x1, y1 = max(0, int(bbox.xmin * w) - padding), max(0, int(bbox.ymin * h) - padding)
        x2, y2 = min(w, int((bbox.xmin + bbox.width) * w) + padding), min(h, int((bbox.ymin + bbox.height) * h) + padding)
        return frame[y1:y2, x1:x2]

    def extract_face_landmarks_468(self, frame):
        results = self.face_mesh.process(frame)
        if not results.multi_face_landmarks: return None
        h, w = frame.shape[:2]
        return np.array([[int(l.x * w), int(l.y * h)] for l in results.multi_face_landmarks[0].landmark])

    def align_face(self, image, landmarks, output_size=(256, 256)):
        left_eye = np.mean(landmarks[33:133], axis=0)
        right_eye = np.mean(landmarks[362:462], axis=0)
        mouth = np.mean(landmarks[78:308], axis=0)
        
        src_pts = np.array([left_eye, right_eye, mouth], dtype=np.float32)
        dst_pts = np.array([
            [output_size[0] * 0.35, output_size[1] * 0.35],
            [output_size[0] * 0.65, output_size[1] * 0.35],
            [output_size[0] * 0.50, output_size[1] * 0.65],
        ], dtype=np.float32)

        M = cv2.getAffineTransform(src_pts, dst_pts)
        return cv2.warpAffine(image, M, output_size, flags=cv2.INTER_LINEAR)

    def mask_eyes(self, frame, landmarks):
        mask = np.ones(frame.shape[:2], dtype=np.uint8)
        for idx in EYE_IDX:
            cv2.circle(mask, tuple(landmarks[idx]), radius=5, color=0, thickness=-1)
        masked_frame = frame.copy()
        masked_frame[mask == 0] = 0
        return masked_frame

    def normalize_color_lab(self, src_img, ref_img):
        src_lab = cv2.cvtColor(src_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        s_m, s_s = cv2.meanStdDev(src_lab)
        r_m, r_s = cv2.meanStdDev(ref_lab)
        src_lab = (src_lab - s_m.flatten()) / (s_s.flatten() + 1e-6)
        src_lab = src_lab * r_s.flatten() + r_m.flatten()
        return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    def calculate_fid(self, faces1, faces2):
        lms1 = [self.extract_face_landmarks_468(f) for f in faces1]
        lms2 = [self.extract_face_landmarks_468(f) for f in faces2]
        
        af1 = [self.align_face(f, l) for f, l in zip(faces1, lms1) if l is not None]
        af2 = [self.align_face(f, l) for f, l in zip(faces2, lms2) if l is not None]
        
        if not af1 or not af2: return float('inf')
        
        ref = af1[0]
        af1 = [self.normalize_color_lab(f, ref) for f in af1]
        af2 = [self.normalize_color_lab(f, ref) for f in af2]

        transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((299, 299)),
            transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        t1 = torch.cat([transform(f).unsqueeze(0) for f in af1]).to(self.device)
        t2 = torch.cat([transform(f).unsqueeze(0) for f in af2]).to(self.device)

        with torch.no_grad():
            feat1 = self.inception_model(t1).cpu().numpy()
            feat2 = self.inception_model(t2).cpu().numpy()

        mu1, sigma1 = feat1.mean(axis=0), np.cov(feat1, rowvar=False)
        mu2, sigma2 = feat2.mean(axis=0), np.cov(feat2, rowvar=False)
        diff = mu1 - mu2
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean): covmean = covmean.real
        
        return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    def calculate_metrics(self, faces1, faces2):
        scores = {"SSIM": [], "PSNR": [], "LPIPS": [], "Landmark": []}
        min_len = min(len(faces1), len(faces2))
        
        lpips_trans = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((256, 256)),
            transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        for i in range(min_len):
            lm1, lm2 = self.extract_face_landmarks_468(faces1[i]), self.extract_face_landmarks_468(faces2[i])
            if lm1 is not None and lm2 is not None:
                # Aligned SSIM
                f1_a, f2_a = self.align_face(faces1[i], lm1), self.align_face(faces2[i], lm2)
                scores["SSIM"].append(ssim(cv2.cvtColor(f1_a, cv2.COLOR_RGB2GRAY), 
                                           cv2.cvtColor(f2_a, cv2.COLOR_RGB2GRAY), data_range=255))
                
                # Masked PSNR & LPIPS
                m1, m2 = self.mask_eyes(faces1[i], lm1), self.mask_eyes(faces2[i], lm2)
                scores["PSNR"].append(psnr(m1, m2))
                
                img1, img2 = lpips_trans(m1).unsqueeze(0).to(self.device), lpips_trans(m2).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    scores["LPIPS"].append(self.lpips_model(img1, img2).item())
                
                # Landmark Dist
                l1_sub, l2_sub = np.delete(lm1, EYE_IDX, axis=0), np.delete(lm2, EYE_IDX, axis=0)
                scores["Landmark"].append(np.mean(np.linalg.norm(l1_sub - l2_sub, axis=1)))

        return {k: np.mean(v) if v else (float('inf') if k != "SSIM" else 0.0) for k, v in scores.items()}

    def calculate_statistics(self, results_list):
        if not results_list: return None
        df = pd.DataFrame(results_list)
        metrics = ['FID', 'SSIM', 'PSNR', 'LPIPS', 'Landmark_Distance']
        stats = {}
        for m in metrics:
            if m in df.columns:
                valid = df[m][~np.isinf(df[m])]
                stats[m] = {"mean": valid.mean(), "std": valid.std(), "min": valid.min(), "max": valid.max()}
        return stats

def run_evaluation():
    evaluator = VideoQualityEvaluator(use_gpu=CONFIG["SETTINGS"]["USE_GPU"])
    out_dir = Path(CONFIG["PATHS"]["OUTPUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ref_root = Path(CONFIG["PATHS"]["REFERENCE"])
    methods = {"mfs": Path(CONFIG["PATHS"]["METHOD_A"]), "ours": Path(CONFIG["PATHS"]["METHOD_B"])}
    
    all_data = []

    for category in sorted([f.name for f in ref_root.iterdir() if f.is_dir()]):
        print(f"\n>>> Processing Category: {category}")
        cat_ref = ref_root / category
        
        for m_name, m_path in methods.items():
            cat_comp = m_path / category
            if not cat_comp.exists(): continue
            
            common_videos = sorted(set(p.stem for p in cat_ref.glob("*.mp4")) & set(p.stem for p in cat_comp.glob("*.mp4")))
            
            for video_name in common_videos:
                print(f"  Evaluating {video_name} ({m_name})...")
                f_ref = evaluator.extract_frames(cat_ref / f"{video_name}.mp4")
                f_comp = evaluator.extract_frames(cat_comp / f"{video_name}.mp4")
                
                faces_ref = [evaluator.detect_and_crop_face(f) for f in f_ref]
                faces_ref = [f for f in faces_ref if f is not None]
                faces_comp = [evaluator.detect_and_crop_face(f) for f in f_comp]
                faces_comp = [f for f in faces_comp if f is not None]

                if faces_ref and faces_comp:
                    fid = evaluator.calculate_fid(faces_ref, faces_comp)
                    metrics = evaluator.calculate_metrics(faces_ref, faces_comp)
                    
                    res = {
                        "Category": category, "Video": video_name, "Method": m_name,
                        "FID": fid, "SSIM": metrics["SSIM"], "PSNR": metrics["PSNR"],
                        "LPIPS": metrics["LPIPS"], "Landmark_Distance": metrics["Landmark"]
                    }
                    all_data.append(res)

    if all_data:
        full_df = pd.DataFrame(all_data)
        full_df.to_csv(out_dir / "all_results_raw.csv", index=False)
        
        for m in ["mfs", "ours"]:
            m_df = full_df[full_df["Method"] == m]
            if not m_df.empty:
                stats = evaluator.calculate_statistics(m_df.to_dict('records'))
                pd.DataFrame(stats).T.to_csv(out_dir / f"summary_{m}.csv")
        
        print(f"\n[!] Done! Results saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    run_evaluation()