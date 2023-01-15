import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skimage

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img')
    parser.add_argument('--img1')
    parser.add_argument('--img2')
    parser.add_argument('--dist', default=False)
    parser.add_argument('--save', default='out.jpg')
    parser.add_argument('--s1', default=0.5)
    parser.add_argument('--s2', default=0.1)
    parser.add_argument('--s3', default=0.4)
    parser.add_argument('--n', default=4)

    return parser.parse_args()

def quantize(img, num):
    (h,w,c) = img.shape
    img2D = img.reshape(h*w,c)
    kmeans_model = KMeans(n_clusters=num)
    cluster_labels = kmeans_model.fit_predict(img2D)
    rgb_cols = kmeans_model.cluster_centers_.round(0).astype(np.uint8)
    img_quant = np.reshape(rgb_cols[cluster_labels],(h,w,c))

    return img_quant, rgb_cols

def merge(path, base, test_path, n, dist, out_path, s1, s2, s3):
    img_org = cv2.imread(path)
    img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img = img_org.copy()

    img_lab = skimage.color.rgb2lab(img)
    lab_dist = np.linalg.norm(img_lab, axis=-1)

    level = (lab_dist / (256 / n)).astype(int)

    offset = np.zeros_like(img_lab)

    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            if (img_lab[h][w][0] < 10):
                offset[h][w] = base[level[h][w]] * s1 ## scale1
            elif (img_lab[h][w][0] > 90):
                offset[h][w] = base[level[h][w]] * s2 ## scale2
            else:
                offset[h][w] = base[level[h][w]] * s3 ## scale3

    img_lab[:, :, 0] = np.clip(img_lab[:, :, 0], 0, 100)
    img_lab[:, :, 1] = np.clip(img_lab[:, :, 1], -127, 128)
    img_lab[:, :, 2] = np.clip(img_lab[:, :, 2], -127, 128)

    img_lab = img_lab + offset * 2

    img = skimage.color.lab2rgb(img_lab)

    for i in range(3):
        img[:, :, i] = np.clip(img[:, :, i], 0, 255)

    test_img = cv2.imread(test_path)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    if (dist):
        dist1 = np.linalg.norm(skimage.color.rgb2lab(img) - skimage.color.rgb2lab(img_org) - np.array([0, 0, 0]))
        dist2 = np.linalg.norm(skimage.color.rgb2lab(img) - skimage.color.rgb2lab(test_img) - np.array([0, 0, 0]))
        
        print(dist1)
        print(dist2)

    plt.subplot(131)
    plt.title("Image1")
    plt.imshow(img_org)
    plt.subplot(132)
    plt.title("Image2")
    plt.imshow(test_img)
    plt.subplot(133)
    plt.title("Image3")
    plt.imshow(img)
    plt.imsave(out_path, img)
    plt.show()

def extract(path1, path2, n):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img1_quant, color1 = quantize(img1, n)
    img2_quant, color2 = quantize(img2, n)

    color1 = sorted(color1, key=lambda x: np.linalg.norm(skimage.color.rgb2lab(x) - np.array([0, 0, 0])))
    color2 = sorted(color2, key=lambda x: np.linalg.norm(skimage.color.rgb2lab(x) - np.array([0, 0, 0])))

    color1 = np.array(color1)
    color2 = np.array(color2)

    color1 = np.array([skimage.color.rgb2lab(color) for color in color1])
    color2 = np.array([skimage.color.rgb2lab(color) for color in color2])

    color_sub = color2 - color1

    return color_sub


def main():
    rt_args = parse_arguments()

    n = int(rt_args.n) ## quantize number

    base = extract(rt_args.img1, rt_args.img2, n)
    merge(rt_args.img, base, rt_args.img2, n, rt_args.dist, rt_args.save, float(rt_args.s1), float(rt_args.s2), float(rt_args.s3))

    return


if __name__ == '__main__':
    main()
