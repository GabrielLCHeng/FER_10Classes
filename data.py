import os
import pandas as pd
import pathlib

temp = pathlib.PosixPath        # linux
# because my laptop is in windows
pathlib.PosixPath = pathlib.WindowsPath


# models data
model_path = './media/models'
fer_models_file = [
    'export_resnet34__acc62_42.pkl', 'export_resnet50_v2_customHead__acc67_20.pkl', 'export_resnet101d__acc45_22.pkl', 
    'export_inceptionv3__acc49_36.pkl', 'export_inceptionvResnetv2__acc48_41.pkl', 'export_efficientnetv2__acc72_93.pkl',

]        # modelName.pth
fer_models_name = ['Resnet34', 'Resnet50', 'Resnet101d', 'Inceptionv3', 'InceptionResnetv2', 'Efficientnetv2']        # modelName

fer_model_dict = {name: os.path.join(model_path, filee) for name, filee in zip(fer_models_name, fer_models_file)}

# fer image data (self provided image)
images_path = './media/test_images'
content_images_name = [
    'angry_1', 'angry_2', 'angry_3', 'angry_4', 'angry_5',
    'boredom_1', 'boredom_2', 'boredom_3', 'boredom_4', 'boredom_5',
    'confused_1', 'confused_2', 'confused_3', 'confused_4', 'confused_5',
    'disgust_1', 'disgust_2', 'disgust_3', 'disgust_4', 'disgust_5',
    'fear_1', 'fear_2', 'fear_3', 'fear_4', 'fear_5',
    'happiness_1', 'happiness_2', 'happiness_3', 'happiness_4', 'happiness_5',
    'neutral_1', 'neutral_2', 'neutral_3', 'neutral_4', 'neutral_5',
    'sadness_1', 'sadness_2', 'sadness_3', 'sadness_4', 'sadness_5',
    'surprised_1', 'surprised_2', 'surprised_3', 'surprised_4', 'surprised_5',
    'thinking_1', 'thinking_2', 'thinking_3', 'thinking_4', 'thinking_5',
]
content_images_file = [
    'angry_test_00009.jpg', 'angry_test_00034.jpg', 'angry_test_00042.jpg', 'angry_test_00045.jpg', 'angry_test_00048.jpg', 
    'boredom_test_00019.jpg', 'boredom_test_00020.jpg', 'boredom_test_00021.jpg', 'boredom_test_00022.jpg', 'boredom_test_00023jpg', 
    'confused_test_00010.jpg', 'confused_test_00011.jpg', 'confused_test_00012.jpg', 'confused_test_00013.jpg', 'confused_test_00014.jpg', 
    'disgust_test_00010.jpg', 'disgust_test_00011.jpg', 'disgust_test_00031.jpg', 'disgust_test_00035.jpg', 'disgust_test_00036.jpg', 
    'fear_test_00025.jpg', 'fear_test_00029.jpg', 'fear_test_00031.jpg', 'fear_test_00043.jpg', 'fear_test_00062.jpg', 
    'happiness_test_00004.jpg', 'happiness_test_00008.jpg', 'happiness_test_00033.jpg', 'happiness_test_00034.jpg', 'happiness_test_00035.jpg', 
    'neutral_test_00001.jpg', 'neutral_test_00019.jpg', 'neutral_test_00021.jpg', 'neutral_test_00024.jpg', 'neutral_test_00044.jpg', 
    'sadness_test_00016.jpg', 'sadness_test_00028.jpg', 'sadness_test_00031.jpg', 'sadness_test_00032.jpg', 'sadness_test_00049.jpg', 
    'surprised_test_00006.jpg', 'surprised_test_00033.jpg', 'surprised_test_00035.jpg', 'surprised_test_00046.jpg', 'surprised_test_00049.jpg', 
    'thinking_test_00010.jpg',  'thinking_test_00011.jpg', 'thinking_test_00012.jpg', 'thinking_test_00013.jpg', 'thinking_test_00014.jpg', 
]    # file_name

content_images_dict = {name: os.path.join(images_path, filee) for name, filee in zip(content_images_name, content_images_file)}