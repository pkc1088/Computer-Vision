import os

import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.

from common import read_img, save_img


def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    # (a) 이미지 이동 (u, v)
    shifted_image = np.roll(image, shift=(v, u), axis=(0, 1))
    # (b) 차이값(SSD: Sum of Squared Differences) 계산
    difference = image - shifted_image
    squared_difference = np.square(difference)
    # (c) 윈도우 내에서 합산 (Zero Padding 사용)
    kernel = np.ones(window_size)  # 윈도우 크기의 평균 필터
    result = scipy.ndimage.convolve(squared_difference, kernel, mode='constant')

    return result


def harris_detector(image, window_size=(5, 5)):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: a image of size H x W
    """
    # compute the derivatives
    # 1️⃣ Sobel 필터로 x, y 방향 미분 계산
    Ix = scipy.ndimage.sobel(image, axis=1, mode='constant')  # x 방향 기울기
    Iy = scipy.ndimage.sobel(image, axis=0, mode='constant')  # y 방향 기울기

    # 2️⃣ 구조 텐서 M의 요소 계산
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # For each image location, construct the structure tensor and calculate
    # the Harris response
    # 3️⃣ 윈도우 내에서 합산 (가우시안 블러를 통한 평균화 효과)
    window = np.ones(window_size)  # 윈도우 필터 (단순 평균)
    Sxx = scipy.ndimage.convolve(Ixx, window, mode='constant')  # Σ I_x^2
    Syy = scipy.ndimage.convolve(Iyy, window, mode='constant')  # Σ I_y^2
    Sxy = scipy.ndimage.convolve(Ixy, window, mode='constant')  # Σ I_x I_y

    # 4️⃣ Harris 응답 점수 계산 (R = det(M) - k * trace(M)^2)
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    k = 0.04 # k 값은 Harris 응답 계산에서 코너 감도 조절 역할 보통 k=0.04 ~ 0.06 사이의 값을 사용
    response = det_M - k * (trace_M ** 2)

    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")


    # -- TODO Task 6: Corner Score --
    W = (5, 5)
    # Computing the corner scores for various u, v values.
    score = corner_score(img, 0, 5, W)
    save_img(score, "./feature_detection/corner_score05.png")
    score = corner_score(img, 0, -5, W)
    save_img(score, "./feature_detection/corner_score0-5.png")
    score = corner_score(img, 5, 0, W)
    save_img(score, "./feature_detection/corner_score50.png")
    score = corner_score(img, -5, 0, W)
    save_img(score, "./feature_detection/corner_score-50.png")



    # # -- TODO Task 7: Harris Corner Detector --
    # (b)
    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")


if __name__ == "__main__":
    main()
