import cv2
import random
import numpy as np
from scipy.linalg import orth


def uint2single(img):
    return np.float32(img / 255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.).round())

def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:  # add color Gaussian noise
        img += np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:  # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:  # add  noise
        L = noise_level2 / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img

def add_JPEG_noise(img):
    quality_factor = random.randint(30, 95)
    img = single2uint(img)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = uint2single(img)
    return img

def random_crop(image, crop_size):
    h, w, c = image.shape
    if h < crop_size or w < crop_size:
        scale = crop_size / min(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = new_h, new_w

    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)
    return image[top: top + crop_size, left: left + crop_size, :]

def synthesize(img_gt: np.ndarray, img_depth: np.ndarray):
    img_depth = (img_depth - img_depth.min()) / (img_depth.max() - img_depth.min())

    beta = np.random.rand(1) * (1.5 - 0.3) + 0.3
    t = np.exp(-(1 - img_depth) * 2.0 * beta)
    t = t[:, :, np.newaxis]

    A = np.random.rand(1) * (1.0 - 0.25) + 0.25
    A_random = np.random.rand(3) * (0.025 - (-0.025)) + (-0.025)
    A = A + A_random

    img_lq = img_gt.copy()
    # adjust luminance
    if np.random.rand(1) < 0.5:
        img_lq = np.power(img_lq, np.random.rand(1) * 1.5 + 1.5)
    # add gaussian noise
    if np.random.rand(1) < 0.5:
        img_lq = add_Gaussian_noise(img_lq)

    # add haze
    img_lq = img_lq * t + A * (1 - t)

    # add JPEG noise
    if np.random.rand(1) < 0.5:
        img_lq = add_JPEG_noise(img_lq)

    if img_gt.shape[-1] > 3:
        img_gt = img_gt[:, :, :3]
        img_lq = img_lq[:, :, :3]

    # img_lq here: [0, 1], HWC, BGR
    img_lq = single2uint(img_lq)
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
    return img_lq