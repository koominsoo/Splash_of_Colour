import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy
from mpl_toolkits.mplot3d import Axes3D
import sys
import cv2


def gaussian_pyramid(img, num_levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr

def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1

    laplacian_pyr = [laplacian_top]
    for i in range(num_levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr

def blend(laplacian_A, laplacian_B, mask_pyr):
    LS = []
    for la, lb, mask in zip(laplacian_A, laplacian_B, mask_pyr):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS

def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i + 1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst

def assignment(df, centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df
def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

def YCbCr_separation(image):
    YCbCr_image = []
    for i in range(3):
        YCbCr_channel = []
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                YCbCr_channel.append(image[j, k, i])
        YCbCr_image.append(YCbCr_channel)
    return np.array(YCbCr_image)

def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    lpA  = [gpA[num_levels-1]]
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1])

    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_

mypath = ['./SoC_images', './LPB_images']
f = []
inputK1 = input('Splash of Colours (0) or Laplacian Pyramid Blending (1) ? ')
if inputK1 is '0':
    mypath = mypath[int(inputK1)]
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        f.extend(filenames)
        break
    for i in range(len(f)):
        print(str(i+1) + '. ' + f[i])
    inputK = input("which photograph to perform splash of Colour? ")
    inputK = int(inputK)
    image_path = os.path.join(mypath, f[inputK - 1])
    image = Image.open(image_path)
    print("Opening " + image_path)
    image_rgb_arr = np.array(image)
    image_ycbcr = image.convert('YCbCr')
    image_ycbcr_arr = np.ndarray((image.size[1], image.size[0], 3), 'u1', image_ycbcr.tobytes())
    print("Image size: " + str(image_ycbcr_arr.shape))
    inputK2 = input("Binary (0) or Multiple Clusters (May not work for low numbers, 1)? ")
    if inputK2 == '1':
        k = input("How many clusters for Colour? ")
        k1 = input("How many clusters for Region? ")
        k = int(k)
        k1 = int(k1)
    else:
        k = 2
    print('Start converting RGB to YCbCr ...')
    '''
    for i in range(image_rgb_arr.shape[0]):
        for j in range(image_rgb_arr.shape[1]):
            R = image_rgb_arr[i][j][0]
            G = image_rgb_arr[i][j][1]
            B = image_rgb_arr[i][j][2]
            image_ycbcr_arr[i][j][0] = 16 + (((R << 6) + (R << 1) + (G << 7) + G + (B << 4) + (B << 3) + B) >> 8)
            image_ycbcr_arr[i][j][1] = 128 + ((-1 * ((R << 5) + (R << 2) + (R << 1))- ((G << 6) + (G << 3) + (G << 1)) + (B << 7) - (B << 4)) >> 8)
            image_ycbcr_arr[i][j][2] = 128 + (((R << 7) - (R << 4) - ((G << 6) + (G << 5) - (G << 1)) - ((B << 4) + (B << 1))) >> 8)
        print('Converting RGB to YCbCr..    Current Row: ' + str(i))
    '''
    fig = plt.figure()
    image_ycbcr_arr_separated = YCbCr_separation(image_ycbcr_arr)
    df = pd.DataFrame({
        'x' : image_ycbcr_arr_separated[1],
        'y' : image_ycbcr_arr_separated[2]
    })
    print('Successfully Converted RGB to YCbCr!')
    print("Start K-means Clustering for Colour...")
    np.random.seed(200)
    centroids = {
        i + 1: [np.random.randint(0, 255), np.random.randint(0, 255)]
        for i in range(k)
    }
    colmap = {
        i + 1: str(i + 1)
        for i in range(k)
    }
    df = assignment(df, centroids)

    while True:
        closest_centroids = df['closest'].copy(deep=True)
        update(centroids)
        print(centroids)
        df = assignment(df, centroids)
        if closest_centroids.equals(df['closest']):
            break
    print(df['color'])
    if inputK2 == '0':
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        for i in range(2):
            temp = np.array(image)
            plt.subplot(1, 3, i + 2)
            conditionK = str(i + 1)
            for j in range(image_rgb_arr.shape[0]):
                for m in range(image_rgb_arr.shape[1]):
                    if df['color'][j * image_rgb_arr.shape[0] + m] != conditionK:
                        R = image_rgb_arr[j][m][0]
                        G = image_rgb_arr[j][m][1]
                        B = image_rgb_arr[j][m][2]
                        for l in range(3):
                            image_rgb_arr[j][m][l] = 0.299 * R + 0.587 * G + 0.114 * B
            plt.imshow(image_rgb_arr)
        plt.show()
    else:
        x_clus = np.zeros(k, dtype='float')
        y_clus = np.zeros(k, dtype='float')
        cluster_cnt = np.zeros((k))
        arrK = np.zeros(k)
        for i in range(k):
            temp = np.array(image)
            conditionK = str(i + 1)
            for j in range(image_ycbcr_arr.shape[0]):
                for m in range(image_ycbcr_arr.shape[1]):
                    if df['color'][j * image_ycbcr_arr.shape[0] + m] == conditionK:
                        arrK[i] += 1
                        x_clus[i] += j
                        y_clus[i] += m
                        cluster_cnt[i] += 1
        for i in range(k):
            x_clus[i] /= cluster_cnt[i]
            y_clus[i] /= cluster_cnt[i]

        df2 = pd.DataFrame({
            'x': x_clus,
            'y': y_clus
        })
        centroids2 = {
            i + 1: [np.random.randint(0, max(df2['x'])), np.random.randint(0, max(df2['y']))]
            for i in range(k1)
        }
        colmap2 = {
            i + 1: str(i + 1)
            for i in range(k1)
        }

        df2 = assignment(df2, centroids2, colmap2)
        iterK = 0
        print("Start K-means Clustering for Region...")
        while True:
            closest_centroids2 = df2['closest'].copy(deep=True)
            centroids2 = update(df2, centroids2)
            df2 = assignment(df2, centroids2, colmap2)
            if closest_centroids2.equals(df2['closest']):
                break
            iterK += 1
            if iterK % 10 == 0:
                print("Current iteration: " + str(iterK))
        arrK2 = np.zeros(k1)
        for i in range(k1):
            temp = np.array(image)
            plt.subplot(2, 3, i + 1)
            conditionK = str(i + 1)
            for j in range(image_ycbcr_arr.shape[0]):
                for m in range(image_ycbcr_arr.shape[1]):
                    temp_num = int(df2['color'][int(df['color'][j * image_ycbcr_arr.shape[0] + m]) - 1])
                    if temp_num == conditionK:
                        arrK2[i] += 1
                        R = temp[j][m][0]
                        G = temp[j][m][1]
                        B = temp[j][m][2]
                        for l in range(3):
                            temp[j][m][l] = 0.299 * R + 0.587 * G + 0.114 * B
            plt.imshow(temp)
            print("Image " + str(i + 1) + " recorded.")
        plt.show()

elif inputK1 is '1':
    A = cv2.imread("./LPB_images/baby.jpg", 0)
    B = cv2.imread("./LPB_images/dog.jpg", 0)
    mask = cv2.imread("./LPB_images/mask.png")
    mask = np.array(mask)
    print(mask.shape)
    mask_array = np.zeros((mask.shape[0], mask.shape[1]), dtype="float32")
    x = 0
    y = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j][0] != 0:
                mask_array[i][j] = 1
    num_levels = 100
    gaussian_pyr_1 = gaussian_pyramid(A, num_levels)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)
    gaussian_pyr_2 = gaussian_pyramid(B, num_levels)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)
    mask_pyr_final = gaussian_pyramid(mask_array, num_levels)
    mask_pyr_final.reverse()
    add_laplace = blend(laplacian_pyr_1,laplacian_pyr_2,mask_pyr_final)
    final  = reconstruct(add_laplace)
    plt.imshow(final[num_levels])
    plt.show()