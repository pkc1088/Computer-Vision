import os

import numpy as np

from common import read_img, save_img
from scipy.ndimage import convolve as scipy_convolve
from matplotlib import pyplot as plt
from PIL import Image


def image_patches(image, patch_size=(16, 16)):
    """
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.
    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    # TODO: Use slicing to complete the function
    output = []
    H, W = image.shape
    M, N = patch_size
    for i in range(0, H, M):
        for j in range(0, W, N):
            patch = image[i:i+M, j:j+N]
            if patch.shape == (M, N): 
                # 패치 정규화 (평균 0, 분산 1)
                patch_normalized = (patch - np.mean(patch)) / np.std(patch)
                output.append(patch_normalized)

    return output




def convolve(image, kernel):
    """
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.
    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    H, W = image.shape
    h, w = kernel.shape
    # 커널을 뒤집어서 컨볼루션 연산 수행 
    flipped_kernel = np.flip(kernel)
    # Zero-padding 적용 
    pad_h = h // 2
    pad_w = w // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    # 출력 이미지 초기화
    convolved_image = np.zeros((H, W))

    # 컨볼루션 연산 수행
    for i in range(H):
        for j in range(W):
            convolved_image[i, j] = np.sum(padded_image[i:i+h, j:j+w] * flipped_kernel)

    return convolved_image

def edge_detection(image):
    """
    Return Ix, Iy and the gradient magnitude of the input image
    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    kx = np.array([[-1, 0, 1]])  # (1, 3)
    ky = np.array([[-1], [0], [1]])  # (3, 1)

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(Ix**2 + Iy**2)

    return Ix, Iy, grad_magnitude

def gaussian_kernel(size=3, sigma=0.572):
    """
    2D Gaussian 커널을 생성합니다.
    Input:
        - size: 커널 크기 (기본값 3x3)
        - sigma: 표준 편차
    Output:
        - kernel: 2D Gaussian 필터
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # 합이 1이 되도록 정규화
    
    return kernel




def sobel_operator(image):
    """
    Return Gx, Gy, and the gradient magnitude.
    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    Sx = np.array([[1, 0, -1], 
                   [2, 0, -2], 
                   [1, 0, -1]])
    Sy = np.array([[1, 2, 1], 
                   [0, 0, 0], 
                   [-1, -2, -1]])
    # TODO: Use convolve() to complete the function
    # Sobel 필터 적용
    Gx = convolve(image, Sx)
    Gy = convolve(image, Sy)
    # Gradient Magnitude 계산
    grad_magnitude = np.sqrt(Gx**2 + Gy**2)
    return Gx, Gy, grad_magnitude




def main():
    # The main function
    img = read_img('./grace_hopper.png')


    # """ Image Patches """
    # if not os.path.exists("./image_patches"):
    #     os.makedirs("./image_patches")
    # # -- TODO Task 1: Image Patches --
    # # First complete image_patches()
    # patches = image_patches(img)
    # # Now choose any three patches and save them
    # chosen_patches = np.vstack([patches[0], patches[1], patches[2]])
    # # chosen_patches should have those patches stacked vertically/horizontally
    # save_img(chosen_patches, "./image_patches/q1_patch.png")
    
    

    # """ Convolution and Gaussian Filter """
    # if not os.path.exists("./gaussian_filter"):
    #     os.makedirs("./gaussian_filter")
    # # -- TODO Task 2: Convolution and Gaussian Filter --
    # # Gaussian 필터 생성
    # kernel_gaussian = gaussian_kernel(size=3, sigma=0.572)
    # # Gaussian 필터 적용
    # filtered_gaussian = convolve(img, kernel_gaussian)
    # save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")
    
    # _, _, edge_detect = edge_detection(img)
    # save_img(edge_detect, "./gaussian_filter/q3_edge.png")

    # _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    # save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")
    # print("Gaussian Filter is done. ")



    # # -- TODO Task 3: Sobel Operator --
    # if not os.path.exists("./sobel_operator"):
    #     os.makedirs("./sobel_operator")
    # Gx, Gy, edge_sobel = sobel_operator(img)
    # save_img(Gx, "./sobel_operator/q2_Gx.png")
    # save_img(Gy, "./sobel_operator/q2_Gy.png")
    # save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")
    # print("Sobel Operator is done. ")




    # -- TODO Task 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    filtered_LoG1 = convolve(img, kernel_LoG1) #None
    filtered_LoG2 = convolve(img, kernel_LoG2) #None 에서 추가함
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # (b)
    # Follow instructions in pdf to approximate LoG with a DoG
    print("LoG Filter is done. ")


if __name__ == "__main__":
    main()
