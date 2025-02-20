import numpy as np
import seaborn as sns
from PIL import Image
import os
import io
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

def get_pdf(data, sampling_point=200):
    # data : (packet_len, subcarriers)
    pdf = np.zeros((sampling_point, data.shape[1]))
    for i in range(data.shape[1]):
        kde = KernelDensity(kernel='gaussian', bandwidth=1.5)
        kde.fit(data[:, i][:, np.newaxis])
        x = np.linspace(0, 50, sampling_point)[:, np.newaxis]
        log_dens = kde.score_samples(x)
        log_dens = np.exp(log_dens)
        pdf[:, i] = log_dens
    return pdf


# clean data, 0.125s segmentation
# Seg CSII: (4 + 9 + 16 + 4, time_len, subcarriers)
# 4: data aug1, 9: priori, 16: event, 4: data aug2
# Format of DADP stream: (10, 3, 171, 128)
def sigment_data_to_RGB_pdf(slice_data, Tw, s_step):
    # sliced len = 0.25s
    # input is amplitude
    # slice data
    # Tw: 0.75, 1, 1.25
    # s_step: 0.125, 0.25, 0.5
    
    # (silice_num, packet_len, subcarriers)
    # transform it into silice PDF
    win_len = int(Tw / 0.125)
    frame_len = int(2 / s_step)
    sliced_num = int((0.125 *  (4 + 9 + 16 + 4) - Tw)/ s_step)
    print(win_len, frame_len, sliced_num)
    pdf = [get_pdf(np.concatenate(slice_data[i:i+win_len], axis=0)) for i in range(sliced_num)]
    pdf = np.array(pdf)# (silice_num, sampling_point, subcarriers)
    #transform to RGB
    rgb_pdf = np.zeros((pdf.shape[0], 3, 171, 128))

    for i, ele in enumerate(pdf):
        heatmap = sns.heatmap(ele.T, cmap='jet', cbar=False)
        ax = plt.gca()
        ax.set_axis_off()
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        image_data = io.BytesIO()
        plt.savefig(image_data, format='png', bbox_inches='tight', pad_inches=0)
        image_data.seek(0)
        img = Image.open(image_data)
        resized_img = img.resize((128, 171))
        rgb_array = np.array(resized_img)[:, :, :3]
        rgb_pdf[i] = np.transpose(rgb_array, (2, 0, 1))
        plt.close()
    return rgb_pdf


def sigment_data_to_RGB_pdf_non_fall(slice_data, Tw, s_step):
    # sliced len = 0.25s
    # input is amplitude
    # slice data
    # Tw: 0.75, 1, 1.25
    # s_step: 0.125, 0.25, 0.5
    
    # (silice_num, packet_len, subcarriers)
    # transform it into silice PDF
    win_len = int(Tw / 0.125)
    frame_len = int(2 / s_step)
    pdf = [get_pdf(np.concatenate(slice_data[i:i+win_len], axis=0)) for i in range(len(slice_data)-win_len)]
    pdf = np.array(pdf)# (silice_num, sampling_point, subcarriers)

    #transform to RGB
    rgb_pdf = np.zeros((pdf.shape[0], 3, 171, 128))
    for i, ele in enumerate(pdf):
        heatmap = sns.heatmap(ele.T, cmap='jet', cbar=False)
        ax = plt.gca()
        ax.set_axis_off()
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        image_data = io.BytesIO()
        plt.savefig(image_data, format='png', bbox_inches='tight', pad_inches=0)
        image_data.seek(0)
        img = Image.open(image_data)
        resized_img = img.resize((128, 171))
        rgb_array = np.array(resized_img)[:, :, :3]
        rgb_pdf[i] = np.transpose(rgb_array, (2, 0, 1))
        plt.close()
    new_rgb_pdf = []
    print('rgb_pdf:     ', rgb_pdf.shape)
    for i in range(len(rgb_pdf)):
        if (i+1)%frame_len == 0 and (i+1+frame_len)<= len(rgb_pdf):
            new_rgb_pdf.append(rgb_pdf[i:i+frame_len])
    new_rgb_pdf = np.array(new_rgb_pdf)

    return new_rgb_pdf


def extract_amp_csi(folder_path):
    dat_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    amp_list = []
    for item in dat_files:
        file = os.path.join(folder_path, item)
        data = np.load(file, allow_pickle=True)
        sliced_fall_data_list = [data[f'arr_{i}'] for i in range(len(data.files))]
        for sliced_fall_data in sliced_fall_data_list:
            amp_list.append(sliced_fall_data)
    return amp_list

# Example:
env = 'home'
user = 'user1'
file_name = ['DAPD_1_125', 'DAPD_1_25', 'DAPD_1_5', 'DAPD_075_25', 'DAPD_125_25']
parm = [(1.25, 0.25)]
motion = 'non_fall'
read_npz_path = f'/data//{env}/{user}/{motion}'
amp_list = extract_amp_csi(read_npz_path)

for ii in range(len(file_name)):
    save_path = f'/data/user/{file_name[ii]}/{env}/{user}/{motion}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for iii in range(len(amp_list)):
        sliced_fall_data = amp_list[iii]
        Tw = parm[ii][0]
        s_step = parm[ii][1]
        rgb_pdf = sigment_data_to_RGB_pdf_non_fall(sliced_fall_data, Tw, s_step)
        print(rgb_pdf.shape)
        np.save(save_path + f'/non_fall{iii}.npy', rgb_pdf)
