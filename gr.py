import gradio as gr
import data as Data
import model as Model
import argparse
import core.logger as Logger
import core.metrics as Metrics
import os
import numpy as np
from PIL import Image

def question_answer(image):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)


    # Path to the directory
    path = '/path/to/your/directory'

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Create or open an image
    image = Image.new('RGB', (100, 100), 'red') # Example: creating a red square image

    # Save the image in the specified directory
    image_path = os.path.join(path, 'image.jpg')
    image.save(image_path)

    input_image = {
        "name": "input_image",
        "mode": "LRHR",
        "dataroot": "dataset/input_image",
        "datatype": "img",
        "l_resolution": 64,
        "r_resolution": 512,
        "data_len": 50
    }

    val_set = Data.create_dataset(input_image, "val")
    val_loader = Data.create_dataloader(
        val_set, input_image, "val")
            
    diffusion = Model.create_model(opt)

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            Metrics.save_img(
                sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

        Metrics.save_img(
            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
        Metrics.save_img(
            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

    # print('{}/{}_{}_inf.png'.format(result_path, current_step, idx))
    return '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx) #np.array(sr_img.tolist())

gr.Interface(fn=question_answer, inputs=["image"], outputs=["image"]).launch()
