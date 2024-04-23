import os
import random
import numpy as np
import SimpleITK as sitk


def file_name_path(file_dir, dir=True, file=False):
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            return dirs
        if len(files) and file:
            return files


def data_split(full_list, ratio, shuffle=True):

    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


def normalize(slice, bottom=99, down=1):
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        tmp[tmp == tmp.min()] = -9
        return tmp


def crop_ceter(img,croph,cropw):
    height, width = img[0].shape
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)
    return img[:, starth:starth+croph, startw:startw+cropw]


if __name__ == '__main__':

    flair_name = "_flair.nii.gz"
    t1_name = "_t1.nii.gz"
    t1ce_name = "_t1ce.nii.gz"
    t2_name = "_t2.nii.gz"
    mask_name = "_seg.nii.gz"

    train_data_label = True
    test_data_label = True

    brats_path = "/21TB/yk/data/BraTS2021_Training_Data"
    trainImage = "/21TB/yk/data/data_prepare_2021/trainImage"
    trainMask = "/21TB/yk/data/data_prepare_2021/trainMask"
    testImage = "/21TB/yk/data/data_prepare_2021/testImage"
    testMask = "/21TB/yk/data/data_prepare_2021/testMask"
    path_list = file_name_path(brats_path)
    path_train_list, path_test_list = data_split(path_list, ratio=0.8, shuffle=True)
    if not os.path.exists(trainImage):
        os.mkdir(trainImage)
    if not os.path.exists(trainMask):
        os.mkdir(trainMask)
    if not os.path.exists(testImage):
        os.mkdir(testImage)
    if not os.path.exists(testMask):
        os.mkdir(testMask)
    if train_data_label:
        for subsetindex in range(len(path_train_list)):
            brats_subset_path = brats_path + "/" + str(path_train_list[subsetindex]) + "/"
            flair_image = brats_subset_path + str(path_train_list[subsetindex]) + flair_name
            t1_image = brats_subset_path + str(path_train_list[subsetindex]) + t1_name
            t1ce_image = brats_subset_path + str(path_train_list[subsetindex]) + t1ce_name
            t2_image = brats_subset_path + str(path_train_list[subsetindex]) + t2_name
            mask_image = brats_subset_path + str(path_train_list[subsetindex]) + mask_name
            flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
            t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
            t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
            t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
            mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
            flair_array = sitk.GetArrayFromImage(flair_src)
            t1_array = sitk.GetArrayFromImage(t1_src)
            t1ce_array = sitk.GetArrayFromImage(t1ce_src)
            t2_array = sitk.GetArrayFromImage(t2_src)
            mask_array = sitk.GetArrayFromImage(mask)
            flair_array_nor = normalize(flair_array)
            t1_array_nor = normalize(t1_array)
            t1ce_array_nor = normalize(t1ce_array)
            t2_array_nor = normalize(t2_array)
            flair_crop = crop_ceter(flair_array_nor, 160, 160)
            t1_crop = crop_ceter(t1_array_nor, 160, 160)
            t1ce_crop = crop_ceter(t1ce_array_nor, 160, 160)
            t2_crop = crop_ceter(t2_array_nor, 160, 160)
            mask_crop = crop_ceter(mask_array, 160, 160)
            print(str(path_train_list[subsetindex]))
            for n_slice in range(flair_crop.shape[0]):
                if np.max(mask_crop[n_slice, :, :]) != 0:
                    maskImg = mask_crop[n_slice, :, :]
                    FourModelImageArray = np.zeros((flair_crop.shape[1], flair_crop.shape[2], 4), np.float64)
                    flairImg = flair_crop[n_slice, :, :]
                    flairImg = flairImg.astype(np.float64)
                    FourModelImageArray[:, :, 0] = flairImg
                    t1Img = t1_crop[n_slice, :, :]
                    t1Img = t1Img.astype(np.float64)
                    FourModelImageArray[:, :, 1] = t1Img
                    t1ceImg = t1ce_crop[n_slice, :, :]
                    t1ceImg = t1ceImg.astype(np.float64)
                    FourModelImageArray[:, :, 2] = t1ceImg
                    t2Img = t2_crop[n_slice, :, :]
                    t2Img = t2Img.astype(np.float64)
                    FourModelImageArray[:, :, 3] = t2Img
                    imagepath = trainImage + "/" + str(path_train_list[subsetindex]) + "_" + str(n_slice) + ".npy"
                    maskpath = trainMask + "/" + str(path_train_list[subsetindex]) + "_" + str(n_slice) + ".npy"
                    np.save(imagepath, FourModelImageArray)
                    np.save(maskpath, maskImg)
        print("Train_data have been prepared！")

    # prepare test data
    if test_data_label:
        for subsetindex in range(len(path_test_list)):
            brats_subset_path = brats_path + "/" + str(path_test_list[subsetindex]) + "/"
            flair_image = brats_subset_path + str(path_test_list[subsetindex]) + flair_name
            t1_image = brats_subset_path + str(path_test_list[subsetindex]) + t1_name
            t1ce_image = brats_subset_path + str(path_test_list[subsetindex]) + t1ce_name
            t2_image = brats_subset_path + str(path_test_list[subsetindex]) + t2_name
            mask_image = brats_subset_path + str(path_test_list[subsetindex]) + mask_name
            flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
            t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
            t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
            t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
            mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
            flair_array = sitk.GetArrayFromImage(flair_src)
            t1_array = sitk.GetArrayFromImage(t1_src)
            t1ce_array = sitk.GetArrayFromImage(t1ce_src)
            t2_array = sitk.GetArrayFromImage(t2_src)
            mask_array = sitk.GetArrayFromImage(mask)
            flair_array_nor = normalize(flair_array)
            t1_array_nor = normalize(t1_array)
            t1ce_array_nor = normalize(t1ce_array)
            t2_array_nor = normalize(t2_array)
            flair_crop = crop_ceter(flair_array_nor, 160, 160)
            t1_crop = crop_ceter(t1_array_nor, 160, 160)
            t1ce_crop = crop_ceter(t1ce_array_nor, 160, 160)
            t2_crop = crop_ceter(t2_array_nor, 160, 160)
            mask_crop = crop_ceter(mask_array, 160, 160)
            print(str(path_test_list[subsetindex]))
            for n_slice in range(flair_crop.shape[0]):
                if np.max(mask_crop[n_slice, :, :]) != 0:
                    maskImg = mask_crop[n_slice, :, :]
                    FourModelImageArray = np.zeros((flair_crop.shape[1], flair_crop.shape[2], 4), np.float64)
                    flairImg = flair_crop[n_slice, :, :]
                    flairImg = flairImg.astype(np.float64)
                    FourModelImageArray[:, :, 0] = flairImg
                    t1Img = t1_crop[n_slice, :, :]
                    t1Img = t1Img.astype(np.float64)
                    FourModelImageArray[:, :, 1] = t1Img
                    t1ceImg = t1ce_crop[n_slice, :, :]
                    t1ceImg = t1ceImg.astype(np.float64)
                    FourModelImageArray[:, :, 2] = t1ceImg
                    t2Img = t2_crop[n_slice, :, :]
                    t2Img = t2Img.astype(np.float64)
                    FourModelImageArray[:, :, 3] = t2Img
                    imagepath = testImage + "/" + str(path_test_list[subsetindex]) + "_" + str(n_slice) + ".npy"
                    maskpath = testMask + "/" + str(path_test_list[subsetindex]) + "_" + str(n_slice) + ".npy"
                    np.save(imagepath, FourModelImageArray)
                    np.save(maskpath, maskImg)
        print("Test_data have been prepared！")
