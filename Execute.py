import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import maximum_filter

# ==========================================
# 1. 설정 및 기본 함수
# ==========================================
IMG_DIR = "./images"
RESIZE_FACTOR = 0.3
np.random.seed(42)

def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def convolve2d_fast(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    shape = (img.shape[0], img.shape[1], kh, kw)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.einsum('ijkl,kl->ij', windows, kernel)

def nms_2d(R, size=7):
    R_max = maximum_filter(R, size=size)
    return (R == R_max) & (R > 0)

# 노이즈 제거를 위한 Adaptive Threshold
def local_percentile_threshold(R, percentile=99.5, grid_rows=4, grid_cols=6):
    H, W = R.shape
    R_th = np.zeros_like(R)
    h_step = H // grid_rows
    w_step = W // grid_cols
    
    global_noise_floor = np.mean(R[R > 0]) * 0.1 
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            y0, y1 = i*h_step, (i+1)*h_step
            x0, x1 = j*w_step, (j+1)*w_step
            block = R[y0:y1, x0:x1]
            valid = (block > global_noise_floor)
            block_pos = block[valid]
            
            if len(block_pos) == 0: continue
            
            thr = np.percentile(block_pos, percentile)
            mask = valid & (block > thr)
            R_th[y0:y1, x0:x1][mask] = block[mask]
    return R_th

def extract_patches(img, corners, patch_size=15):
    r = patch_size // 2
    H, W = img.shape
    patches = []
    valid_xy = []  # 살아남은 좌표를 담을 리스트
    
    for (x, y) in corners:
        # 외곽 경계 체크
        if x-r < 0 or x+r >= W or y-r < 0 or y+r >= H: 
            continue
        
        patches.append(img[y-r:y+r+1, x-r:x+r+1])
        valid_xy.append((x, y))  # 패치를 담을 때 좌표도 같이 담음
        
    return patches, valid_xy

def harris_corners_and_patches(img_gray, patch_size=15, percentile=99.0, nms_size=5):
    img = convolve2d_fast(img_gray, gaussian_kernel(5, 1.0))
    Ix = np.zeros_like(img); Iy = np.zeros_like(img)
    Ix[:, 1:-1] = img[:, 2:] - img[:, :-2]
    Iy[1:-1, :] = img[2:, :] - img[:-2, :]
    
    Ixx = Ix**2; Iyy = Iy**2; Ixy = Ix*Iy
    k_st = gaussian_kernel(5, 1.5)
    Sxx = convolve2d_fast(Ixx, k_st); Syy = convolve2d_fast(Iyy, k_st); Sxy = convolve2d_fast(Ixy, k_st)
    
    k = 0.04
    R = (Sxx*Syy - Sxy**2) - k*(Sxx + Syy)**2
    
    R_th = local_percentile_threshold(R, percentile=percentile)
    corners_mask = nms_2d(R_th, size=nms_size)
    ys, xs = np.where(corners_mask)
    corners = list(zip(xs, ys))
    
    return extract_patches(img, corners, patch_size)

# ==========================================
# 2. 매칭 및 RANSAC (Step 3 & 4)
# ==========================================
def normalize_patch(p):
    v = p.astype(np.float32).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    if n < 1e-6: return None
    return v / n

def patches_to_descriptors_aligned(patches, corners):
    valid_descs = []
    valid_corners = []
    
    for p, c in zip(patches, corners):
        d = normalize_patch(p)
        if d is not None:
            valid_descs.append(d)
            valid_corners.append(c)
            
    return np.array(valid_descs), valid_corners

def ssd_all_to_all(P1, P2):
    a2 = np.sum(P1*P1, axis=1, keepdims=True)
    b2 = np.sum(P2*P2, axis=1, keepdims=True).T
    ab = P1 @ P2.T
    return a2 + b2 - 2*ab

def match_patches_ssd(descs1, descs2, ratio_thr=0.88):
    D = ssd_all_to_all(descs1, descs2)
    matches = []
    for i in range(D.shape[0]):
        idx = np.argsort(D[i])
        j1, j2 = idx[0], idx[1]
        if D[i, j1] / (D[i, j2] + 1e-12) < ratio_thr:
            matches.append((i, j1))
    return matches

def matches_to_points(matches, corners1, corners2):
    pts1, pts2 = [], []
    for i, j in matches:
        pts1.append(corners1[i])
        pts2.append(corners2[j])
    return np.array(pts1), np.array(pts2)

def compute_homography(p0, p1):
    A = []
    for (x, y), (xp, yp) in zip(p0, p1):
        A.append([-x, -y, -1,  0,  0,  0, x*xp, y*xp, xp])
        A.append([ 0,  0,  0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    
    # [안전장치] SVD 수렴 실패 방지
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return np.eye(3)
        
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def reprojection_error(H, p0, p1):
    p0_h = np.hstack([p0, np.ones((len(p0), 1))])
    p1_proj = (H @ p0_h.T).T
    
    # 0으로 나누기 방지
    w = p1_proj[:, 2:3]
    w[np.abs(w) < 1e-10] = 1e-10
    p1_proj /= w
    p1_proj[:, :2] = np.clip(p1_proj[:, :2], -100000, 100000)
    
    return np.linalg.norm(p1_proj[:, :2] - p1, axis=1)

def ransac_homography(pts0, pts1, num_iter=5000, threshold=4.0):
    best_inliers_mask = None
    best_H = None
    max_inlier_count = 0
    N = len(pts0)
    if N < 4: return None, None
    
    for _ in range(num_iter):
        idx = np.random.choice(N, 4, replace=False)
        H = compute_homography(pts0[idx], pts1[idx])
        
        # H가 비정상이면 건너뜀
        if np.any(np.isnan(H)) or np.any(np.isinf(H)): continue
            
        errors = reprojection_error(H, pts0, pts1)
        inliers_mask = errors < threshold
        count = np.sum(inliers_mask)
        
        if count > max_inlier_count:
            max_inlier_count = count
            best_inliers_mask = inliers_mask
            best_H = H
            
    return best_H, best_inliers_mask

# ==========================================
# 3. 워핑 및 블렌딩 유틸리티
# ==========================================
def warp_perspective_safe(img, H_mat, out_shape):
    out_H, out_W = out_shape
    if out_H * out_W > 500000000: # 메모리 ovf 방지
        print(f"[Critical Warning] Canvas too huge ({out_W}x{out_H}). Skipping.")
        return np.zeros((out_H, out_W), dtype=np.float32)

    H_inv = np.linalg.inv(H_mat)
    xs = np.arange(out_W, dtype=np.float32)
    ys = np.arange(out_H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    w_vec = H_inv[2, 0] * xx + H_inv[2, 1] * yy + H_inv[2, 2]
    w_safe = w_vec.copy()
    w_safe[np.abs(w_safe) < 1e-5] = 1.0 

    src_x = (H_inv[0, 0] * xx + H_inv[0, 1] * yy + H_inv[0, 2]) / w_safe
    src_y = (H_inv[1, 0] * xx + H_inv[1, 1] * yy + H_inv[1, 2]) / w_safe

    # NaN / Inf 방지
    invalid_mask = np.isnan(src_x) | np.isinf(src_x) | np.isnan(src_y) | np.isinf(src_y)
    src_x[invalid_mask] = 0
    src_y[invalid_mask] = 0

    h, w = img.shape
    mask = (src_x >= 0) & (src_x < w - 1) & (src_y >= 0) & (src_y < h - 1) & (np.abs(w_vec) > 1e-5) & (~invalid_mask)

    warped = np.zeros((out_H, out_W), dtype=np.float32)
    
    x0 = np.clip(np.floor(src_x).astype(np.int32), 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(np.floor(src_y).astype(np.int32), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    Ia = img[y0, x0]; Ib = img[y1, x0]
    Ic = img[y0, x1]; Id = img[y1, x1]

    wa = (x1 - src_x) * (y1 - src_y)
    wb = (x1 - src_x) * (src_y - y0)
    wc = (src_x - x0) * (y1 - src_y)
    wd = (src_x - x0) * (src_y - y0)

    warped[mask] = (wa*Ia + wb*Ib + wc*Ic + wd*Id)[mask]
    return warped

def create_weight_mask(h, w):
    Y, X = np.indices((h, w))
    dist_x = np.minimum(X, w - 1 - X)
    dist_y = np.minimum(Y, h - 1 - Y)
    weight = np.minimum(dist_x, dist_y).astype(np.float32)
    return weight / (weight.max() + 1e-5)

# ==========================================
# 4. 메인 실행 
# ==========================================
print(f"=== Final Stitching (Resize: {RESIZE_FACTOR}) ===")

all_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png"))])
img_files = all_files[1:10]

images_gray = []

print("Loading images...")
for fname in img_files:
    img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
    
    # 리사이즈
    if RESIZE_FACTOR != 1.0:
        new_w = int(img.width * RESIZE_FACTOR)
        new_h = int(img.height * RESIZE_FACTOR)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
    images_gray.append(np.array(img.convert("L"), dtype=np.float32))

print(f"Loaded {len(images_gray)} images. Size: {images_gray[0].shape}")

# (2) 이웃한 이미지 간 호모그래피 계산
H_neighbors = [None] * (len(images_gray) - 1)

print("\n[Step 1] Computing Pairwise Homographies...")
for i in range(len(images_gray) - 1):
    img1 = images_gray[i]
    img2 = images_gray[i+1]
    
    p1, c1 = harris_corners_and_patches(img1, patch_size=15, percentile=99.0)
    p2, c2 = harris_corners_and_patches(img2, patch_size=15, percentile=99.0)
    
    d1, vc1 = patches_to_descriptors_aligned(p1, c1)
    d2, vc2 = patches_to_descriptors_aligned(p2, c2)
    
    matches = match_patches_ssd(d2, d1, ratio_thr=0.88)
    
    pts_src, pts_dst = matches_to_points(matches, vc2, vc1)
    
    H, inliers = ransac_homography(pts_src, pts_dst, threshold=4.0)
    
    if inliers is not None:
        real_inliers = np.sum(inliers)
        if real_inliers < 10:
             print(f"  Match {i}-{i+1}: {real_inliers} inliers")
             H_neighbors[i] = np.eye(3)
        else:
             print(f"  Match {i}-{i+1}: ({real_inliers}/{len(matches)} inliers)")
             H_neighbors[i] = H
    else:
        print(f"  Match {i}-{i+1}: FAILED -> Using Identity")
        H_neighbors[i] = np.eye(3)

# (3) Global Homography 계산 (가운데 이미지 기준)
print("\n[Step 2] Calculating Global Homographies...")
center_idx = len(images_gray) // 2 
H_global = [None] * len(images_gray)
H_global[center_idx] = np.eye(3)

for i in range(center_idx, len(images_gray) - 1):
    H_global[i+1] = H_global[i] @ H_neighbors[i]

for i in range(center_idx, 0, -1):
    H_inv = np.linalg.inv(H_neighbors[i-1])
    H_global[i-1] = H_global[i] @ H_inv

# (4) 캔버스 크기 계산 및 렌더링
print("\n[Step 3] Rendering Panorama...")
all_corners = []
valid_indices = []

for i in range(len(images_gray)):
    if H_global[i] is None: continue
    valid_indices.append(i)
    h, w = images_gray[i].shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    corners_h = np.concatenate([corners.reshape(-1, 2), np.ones((4, 1))], axis=1)
    corners_proj = (H_global[i] @ corners_h.T).T
    
    div = corners_proj[:, 2:3]
    div[np.abs(div) < 1e-10] = 1.0
    corners_proj = corners_proj[:, :2] / div
    all_corners.append(corners_proj)

all_corners = np.vstack(all_corners)
x_min, y_min = np.min(all_corners, axis=0)
x_max, y_max = np.max(all_corners, axis=0)

PADDING = 100
canvas_w = int(np.ceil(x_max - x_min)) + (PADDING * 2)
canvas_h = int(np.ceil(y_max - y_min)) + (PADDING * 2)

print(f"Final Canvas Size: {canvas_w} x {canvas_h}")

# 캔버스 크기가 너무 크면 중단
if canvas_w * canvas_h > 400000000:
    print("Canvas too large! Stopping to prevent crash.")
else:
    T_shift = np.array([[1, 0, -x_min + PADDING], [0, 1, -y_min + PADDING], [0, 0, 1]])

    panorama_num = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    panorama_den = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for i in valid_indices:
        print(f"  Warping Image {i}...")
        H_final = T_shift @ H_global[i]
        
        warped_img = warp_perspective_safe(images_gray[i], H_final, (canvas_h, canvas_w))
        
        # 빈 이미지면 pass
        if np.max(warped_img) == 0: continue

        h_src, w_src = images_gray[i].shape
        weight_mask = create_weight_mask(h_src, w_src)
        warped_weight = warp_perspective_safe(weight_mask, H_final, (canvas_h, canvas_w))
        
        mask = warped_img > 0
        panorama_num[mask] += (warped_img * warped_weight)[mask]
        panorama_den[mask] += warped_weight[mask]

    # (6) 최종 결과 저장
    panorama_den[panorama_den < 1e-5] = 1.0
    final_panorama = panorama_num / panorama_den
    final_panorama[panorama_num == 0] = 0

    save_path = "./results/final_panorama_generalization.png"
    os.makedirs("./results", exist_ok=True)
    Image.fromarray(np.clip(final_panorama, 0, 255).astype(np.uint8)).save(save_path)
    print(f"\nSaved successfully to: {save_path}")

    plt.figure(figsize=(20, 8))
    plt.imshow(final_panorama, cmap='gray')
    plt.axis('off')
    plt.title(f"Final Result (Resize={RESIZE_FACTOR})")
    plt.show()