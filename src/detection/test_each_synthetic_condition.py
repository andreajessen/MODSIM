import os

light_condition1 = ['2023-05-09_1328_dnv_scenario1_full_00_000_NoonClear']
light_condition2 = ['2023-05-09_1257_dnv_scenario1_full_00_000_AfternoonClear']
light_condition3 = ['2023-05-09_1308_dnv_scenario1_full_00_000_EveningClear']
light_condition4 = ['2023-05-09_1324_dnv_scenario1_full_00_000_NightClear']

cloud_light1 = ['2023-05-09_1330_dnv_scenario1_full_00_000_NoonCloudy']
cloud_light2 = ['2023-05-09_1300_dnv_scenario1_full_00_000_AfternoonCloudy']
cloud_light3 = ['2023-05-09_1311_dnv_scenario1_full_00_000_EveningCloudy']
cloud_light4 = ['2023-05-09_1326_dnv_scenario1_full_00_000_NightCloudy']

rain_light1 = ['2023-05-09_1332_dnv_scenario1_full_00_000_NoonCloudy_Rain']
rain_light2 = ['2023-05-09_1313_dnv_scenario1_full_00_000_EveningCloudy_Rain']
rain_light3 = ['2023-05-09_1302_dnv_scenario1_full_00_000_AfternoonCloudy_Rain']

foggy = ['2023-05-09_1315_dnv_scenario1_full_00_000_FoggyClear']
foggy_cloudy = ['2023-05-09_1317_dnv_scenario1_full_00_000_FoggyCloudy']

stormy = ['2023-05-09_1339_dnv_scenario1_full_00_000_StormClouds']
stormy_rain = ['2023-05-09_1341_dnv_scenario1_full_00_000_StormyClouds_Rain']


all_conditions = [light_condition1, light_condition2, light_condition3, light_condition4, cloud_light1, cloud_light2, cloud_light3, cloud_light4, rain_light1, rain_light2, rain_light3, foggy, foggy_cloudy, stormy, stormy_rain]
all_conditions_names = ['light_condition1', 'light_condition2', 'light_condition3', 'light_condition4', 'cloud_light1', 'cloud_light2', 'cloud_light3', 'cloud_light4', 'rain_light1', 'rain_light2', 'rain_light3', 'foggy', 'foggy_cloudy', 'stormy', 'stormy_rain']


for i, condition in enumerate(all_conditions):
    condition_name = all_conditions_names[i]
    source_path = f"/cluster/home/solveijm/DNV_synthetic_data_w_pose/{condition[0]}/test.txt"
    string = f"yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_50e_1024imgz_mixed_synthetic_and_hurtigruta_correct_split_16_05/weights/best.pt source='{source_path}' imgsz=1024 name=test_induvidual_contions_1024imgz_hurtigruta_and_synthetic_18_05/test_{condition_name} save=True save_conf=True save_txt=True"
    os.system(string)