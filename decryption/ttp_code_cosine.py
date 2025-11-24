import os
import json
import base64
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import unpad
import numpy as np
import sys
import cv2
import torch
from torchvision import transforms
from PIL import Image
import pprint
from tabulate import tabulate
from types import SimpleNamespace
import matplotlib.pyplot as plt
import seaborn as sns
import zlib
import msgpack

# --- Path setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils
from src.backbones import get_model  # Make sure this is valid
# from edgeface.face_alignment import align

# --- Constants ---
KEYS_PATH = "../output/face_keys.msgpack"
PRIVATE_KEY_PATH = "private_key.pem"
DB_PATH = os.path.abspath("../input/ttp_DB")
SIMILARITY_THRESHOLD = 0.65

# --- Load model ---
model = get_model("edgeface_s_gamma_05")
model.load_state_dict(torch.load("../src/models/edgeface_s_gamma_05.pt", map_location='cpu'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Image transform ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Init MediaPipe landmark detector ---
interpreter, input_details, output_details = utils.initialize_landmark_detector()
config = SimpleNamespace()
config.DETECTOR_MODEL = "yunet"

# You can define additional config fields if needed later
detector = utils.initialize_detector(config)


# --- Load Private Key ---
with open(PRIVATE_KEY_PATH, "rb") as key_file:
    private_key = RSA.import_key(key_file.read())
rsa_cipher = PKCS1_OAEP.new(private_key)

def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=1, thickness=1):
    img_with_landmarks = image.copy()
    for entry in landmarks.astype(np.int32):
        cv2.circle(img_with_landmarks, (entry[0], entry[1]), radius, color, thickness)
    return img_with_landmarks

# --- AES Decryption Helper ---
def decrypt_aes_bytes(ciphertext_bytes, key):
    cipher = AES.new(key, AES.MODE_ECB)
    plaintext_padded = cipher.decrypt(ciphertext_bytes)
    try:
        return unpad(plaintext_padded, 16)
    except ValueError:
        return None

def cosine_similarity(vec1, vec2):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (norm1 * norm2)


# --- Decrypt embeddings ---
with open(KEYS_PATH, "rb") as f:
    encrypted_keys = msgpack.unpackb(zlib.decompress(f.read()), raw=False)


face_id_to_aes_key = {}
for face_id, data in encrypted_keys.items():
    if face_id == "video_identifier":
        continue
    encrypted_key_bytes = base64.b64decode(data["aes_key"])
    aes_key = rsa_cipher.decrypt(encrypted_key_bytes)
    decrypted_embedding_bytes = decrypt_aes_bytes(data["embedding"], aes_key)
    decrypted_str = decrypted_embedding_bytes.decode('utf-8')

    face_id_to_aes_key[face_id] = {
        "aes_key": aes_key,
        "embedding": decrypted_str
    }

print("✅ Decryption complete.")

# --- Convert to numpy embeddings ---
face_embeddings = {
    face_id: np.array(json.loads(info["embedding"]), dtype=np.float32)
    for face_id, info in face_id_to_aes_key.items()
}

# --- Group faces ---
visited = set()
groups = {}
group_count = 0

for face_id, emb in face_embeddings.items():
    if face_id in visited:
        continue

    group_key = f"group_{group_count}"
    groups[group_key] = [{"face_id": face_id, "aes_key": face_id_to_aes_key[face_id]["aes_key"]}]
    visited.add(face_id)

    for other_id, other_emb in face_embeddings.items():
        if other_id in visited or other_id == face_id:
            continue

        similarity = cosine_similarity(emb, other_emb)
        print(f"Comparing {face_id} with {other_id}: Similarity = {similarity:.4f}")
        if similarity > SIMILARITY_THRESHOLD:
            visited.add(other_id)
            groups[group_key].append({
                "face_id": other_id,
                "aes_key": face_id_to_aes_key[other_id]["aes_key"]
            })

    group_count += 1

print("✅ Face grouping complete.")
pprint.pprint(groups)

# --- Process DB and extract embeddings ---
db_embeddings = {}

for fname in os.listdir(DB_PATH):
    if not fname.lower().endswith((".jpg", ".jpeg", "png")):
        continue

    person_name = os.path.splitext(fname)[0]
    img_path = os.path.join(DB_PATH, fname)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"⚠️ Could not load image: {img_path}")
        continue

    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = utils.detect_faces(face_rgb, detector, config)
    
    # if not faces:
    #     print(f"❌ No face detected in {fname}")
    #     face_roi = face_rgb  # Use the whole image if no face detected
    # else:                    
    #     face_box = faces[0]["box"]  # Take the first detected face
    #     face_box = utils.expand_box_with_margin(face_box, 0.3, face_rgb.shape[1], face_rgb.shape[0])
    #     x1, y1, x2, y2 = face_box
    #     face_roi = face_rgb[y1:y2, x1:x2]
    face_roi = face_rgb

    # Resize/crop face ROI to model's expected input
    face_crop = cv2.resize(face_roi, (192, 192))
    landmarks = utils.get_landmarks_interpretor(face_crop, interpreter, input_details, output_details)
    landmarks[:, 0] *= 192
    landmarks[:, 1] *= 192

    # # convert to bgr for drawing
    # face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
    # # Draw landmarks on the face crop
    # overlay = draw_landmarks(face_crop, landmarks)
    # # display the landmarks on the face crop
    # cv2.imshow("Landmarks", overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    aligned = utils.align_face(face_crop, landmarks)
    
    Image.fromarray(aligned).show()
    pil_img = Image.fromarray(aligned)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # aligned = align.get_aligned_face(img_path) # align face
    # input_tensor = transform(aligned)
    with torch.no_grad():
        embedding = model(input_tensor).cpu().numpy().flatten()

    db_embeddings[person_name] = embedding

print("✅ DB embeddings computed.")

db_names = list(db_embeddings.keys())
db_vectors = [db_embeddings[name] for name in db_names]
db_vectors = np.stack(db_vectors, axis=0)  # shape: (N, 512)

# Normalize vectors
norm_vectors = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)

# Compute cosine similarity matrix
similarity_matrix = np.dot(norm_vectors, norm_vectors.T)

# Round for display
rounded_matrix = np.round(similarity_matrix, 2)

# Display similarity heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(rounded_matrix, xticklabels=db_names, yticklabels=db_names,
            annot=True, fmt=".2f", cmap='viridis', square=True, cbar_kws={"shrink": 0.8})
plt.title("Cosine Similarity Matrix of DB Embeddings")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Match each group to closest DB person ---
from tabulate import tabulate  # Add this at the top if not already

final_output = {}

for group_key, members in groups.items():
    best_similarity = -1
    best_match = None

    for db_key, db_embedding in db_embeddings.items():
        match_found = False

        for member in members:
            member_embedding = torch.tensor(
                face_embeddings[member["face_id"]],
                dtype=torch.float32
            ).unsqueeze(0)

            similarity = cosine_similarity(member_embedding, torch.tensor(db_embedding).unsqueeze(0)).item()
            print(f"{group_key}:{member['face_id']} -> {db_key}: {similarity:.4f}")

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = db_key

        #     if similarity >= SIMILARITY_THRESHOLD:
        #         match_found = True
        #         break  # No need to check other members if one matches

        # if match_found:
        #     break  # No need to check other db entries if one match is found

    if best_similarity >= SIMILARITY_THRESHOLD:
        print(f"✅ Group {group_key} matched with {best_match} (Similarity: {best_similarity:.4f})")
        final_output[best_match] = {
            "members": [
                {
                    "face_id": m["face_id"],
                    "aes_key": base64.b64encode(m["aes_key"]).decode("utf-8")
                } for m in members
            ],
            "similarity": round(best_similarity, 4)
        }

    else:
        print(f"❌ Group {group_key} has no match in DB (Best similarity: {best_similarity:.4f})")
        final_output[f"{group_key}_no_match"] = {
            "members": [
                {
                    "face_id": m["face_id"],
                    "aes_key": base64.b64encode(m["aes_key"]).decode("utf-8")
                } for m in members
            ],
            "similarity": round(best_similarity, 4)
        }


# Display results in a table
table_data = []
for name, data in final_output.items():
    table_data.append([name, len(data["members"]), data["similarity"]])

print("\n📊 Final Matching Results:\n")
print(tabulate(table_data, headers=["Group/Name", "Num Members", "Similarity"], tablefmt="pretty"))

# save final output as JSON
# This should only be saved in output directory / sent to wearer incase the match is found and the bystander responds in the affirmative.
# For the sake of this demo, we will save it regardless of matching or bystander decision so we can view the restore functionality.
output_path = "../output/final_matching_results.json"
with open(output_path, "w") as f:
    json.dump(final_output, f, indent=4)