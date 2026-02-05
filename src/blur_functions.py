# This file contains functions to apply different types of blurs to a face region in an image.

import cv2
import numpy as np
# from functools import lru_cache

# FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
#                            (17, 314), (314, 405), (405, 321), (321, 375),
#                            (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
#                            (37, 0), (0, 267),
#                            (267, 269), (269, 270), (270, 409), (409, 291),
#                            (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
#                            (14, 317), (317, 402), (402, 318), (318, 324),
#                            (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
#                            (82, 13), (13, 312), (312, 311), (311, 310),
#                            (310, 415), (415, 308)])

# FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
#                                (374, 380), (380, 381), (381, 382), (382, 362),
#                                (263, 466), (466, 388), (388, 387), (387, 386),
#                                (386, 385), (385, 384), (384, 398), (398, 362)])

# FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
#                                    (295, 285), (300, 293), (293, 334),
#                                    (334, 296), (296, 336)])

# FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
#                                 (145, 153), (153, 154), (154, 155), (155, 133),
#                                 (33, 246), (246, 161), (161, 160), (160, 159),
#                                 (159, 158), (158, 157), (157, 173), (173, 133)])

# FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
#                                     (70, 63), (63, 105), (105, 66), (66, 107)])


# FACEMESH_FACE_OVAL = frozenset([(10, 338), (338, 297), (297, 332), (332, 284),
#                                 (284, 251), (251, 389), (389, 356), (356, 454),
#                                 (454, 323), (323, 361), (361, 288), (288, 397),
#                                 (397, 365), (365, 379), (379, 378), (378, 400),
#                                 (400, 377), (377, 152), (152, 148), (148, 176),
#                                 (176, 149), (149, 150), (150, 136), (136, 172),
#                                 (172, 58), (58, 132), (132, 93), (93, 234),
#                                 (234, 127), (127, 162), (162, 21), (21, 54),
#                                 (54, 103), (103, 67), (67, 109), (109, 10)])
# FACEMESH_CONTOURS = frozenset().union(*[
#     FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
#     FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL
# ])

# oval_indices = list(set([pt for pair in FACEMESH_CONTOURS for pt in pair]))

# @lru_cache(maxsize=4)
# def get_gaussian_kernel(ksize, sigma):
#     """Returns a 2D Gaussian kernel."""
#     k1d = cv2.getGaussianKernel(ksize, sigma)
#     return k1d @ k1d.T  # outer product to get 2D kernel

def apply_gaussian_blur(frame, landmarks, ksize=7, sigma=7, scale = 0.1):
    """Applies Gaussian blur to the face region using convex hull."""
    frame_height, frame_width, _ = frame.shape
    # landmarks = np.array(landmarks, dtype=np.int32)
    # Compute convex hull of the landmarks
    hull = cv2.convexHull(landmarks)
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Get bounding box of the convex hull
    x, y, w, h = cv2.boundingRect(hull)
    
    x = max(0, x)
    y = max(0, y)
    roi = frame[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]

    # Get and apply cached Gaussian kernel
    # kernel = get_gaussian_kernel(ksize, sigma)
    # roi_blur = cv2.filter2D(roi, -1, kernel)
    # roi_blur = cv2.GaussianBlur(roi, (ksize, ksize), sigma)
    
    if scale < 1.0:
        ksize = min(max(3, 1 * (min(w, h) // 5) + 1), ksize)
        if ksize % 2 == 0:
            ksize += 1
        roi_small = cv2.resize(roi, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        roi_blur_small = cv2.GaussianBlur(roi_small, (ksize, ksize), sigma)
        roi_blur = cv2.resize(roi_blur_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        ksize = min(max(11, 4 * (min(w, h) // 10) + 1), 201)  # Must be odd
        roi_blur = cv2.GaussianBlur(roi, (ksize, ksize), 201)

    if roi_blur.shape != roi.shape:
        roi_blur = cv2.resize(roi_blur, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask_3ch = cv2.merge([roi_mask] * 3)
    roi = np.where(mask_3ch == 255, roi_blur, roi)
    
    frame[y:y+h, x:x+w] = roi
    return frame

def apply_blur(frame, landmarks, color=(50, 50, 50)):
    """Applies a color blur to the face region using convex hull."""
    frame_height, frame_width = frame.shape[:2]
    
    hull = cv2.convexHull(landmarks)
    x, y, w, h = cv2.boundingRect(hull)
    
    # Ensure the roi is within frame boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame_width - x)
    h = min(h, frame_height - y)
    
    if w <= 0 or h <= 0:
        return frame
    
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    hull_roi = hull - np.array([x, y])
    
    cv2.fillConvexPoly(roi_mask, hull_roi, 255)
    
    roi = frame[y:y+h, x:x+w]
    colored_roi = np.full_like(roi, color, dtype=np.uint8)
    mask_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR) 
    roi_result = np.where(mask_3ch > 0, colored_roi, roi)
    
    frame[y:y+h, x:x+w] = roi_result
    
    return frame


def apply_blur_new(frame, hull, x, y, w, h, color=(78, 99, 150)):
    """Applies a color blur to the face region using convex hull."""
    if w <= 0 or h <= 0:
        return frame
    
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    hull_roi = hull - np.array([x, y])
    
    cv2.fillConvexPoly(roi_mask, hull_roi, 255)
    
    roi = frame[y:y+h, x:x+w]
    colored_roi = np.full_like(roi, color, dtype=np.uint8)
    mask_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR) 
    roi_result = np.where(mask_3ch > 0, colored_roi, roi)
    
    frame[y:y+h, x:x+w] = roi_result
    
    return frame

def apply_ellipse_blur(frame, landmarks):
    frame_height, frame_width, _ = frame.shape
    points = np.array(landmarks, dtype=np.int32)

    # Compute the center of the face (you can also use specific landmarks like nose tip)
    center = tuple(np.mean(points, axis=0).astype(int))

    # Compute approximate width and height of the face region
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    width = int((x_coords.max() - x_coords.min()) * 0.6)
    height = int((y_coords.max() - y_coords.min()) * 0.8)

    # Create the elliptical mask
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    axes = (width // 2, height // 2)  # ellipse radii
    cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

    # Fill masked region with color
    frame[mask > 0] = (50, 50, 50)  # dark gray

    return frame

def apply_box_blur(frame, box, color=(50, 50, 50)):
    """Applies a color blur to the rectangular face region using a bounding box."""
    frame_height, frame_width = frame.shape[:2]
    
    x1, y1, x2, y2 = box

    # Clamp box coordinates to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return frame

    roi = frame[y1:y2, x1:x2]
    colored_roi = np.full_like(roi, color, dtype=np.uint8)

    frame[y1:y2, x1:x2] = colored_roi

    return frame