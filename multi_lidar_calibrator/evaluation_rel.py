import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def calculate_rmse_error(gt_pose, calc_pose):
    gt_translation = np.array(gt_pose[:3])
    calc_translation = np.array(calc_pose[:3])
    translation_errors = gt_translation - calc_translation
    translation_error_rmse = np.sqrt(np.mean(translation_errors ** 2))

    gt_rotation = R.from_euler('xyz', gt_pose[3:], degrees=True)
    calc_rotation = R.from_euler('xyz', calc_pose[3:], degrees=True)
    rotation_error = gt_rotation.inv() * calc_rotation
    rotation_error_degrees = rotation_error.magnitude() * (180.0 / np.pi)
    rotation_errors_individual = rotation_error.as_euler('xyz', degrees=True)

    return translation_errors, translation_error_rmse, rotation_error_degrees, rotation_errors_individual

def evaluate(calibration_results: list, sensors_config_path: str):
    config = load_config(sensors_config_path)
    # Get the name of the lidar we are evaluating
    lidar_name = calibration_results[0]
    # Get the calibration values from the calibration results for the lidar we are evaluating
    calibration_res = calibration_results[1:]
    # Fetch the ground truth calibration values for the lidar from the config file
    ground_truth = config['sensor_kit_calibration']["hesai_center_lidar"][lidar_name]
    # Get the ground truth translation and rotation values
    translation_values_from_config = [ground_truth['x'], ground_truth['y'], ground_truth['z']]
    rotation_values_from_config = [ground_truth['roll'], ground_truth['pitch'], ground_truth['yaw']]
    # Combine the translation and rotation values
    gt_data = translation_values_from_config + rotation_values_from_config

    gt_data_arr = np.array(gt_data)
    translation_errors, translation_error_rmse, rotation_error_degrees, rotation_errors_individual = calculate_rmse_error(gt_data_arr, calibration_res)
    print(f"  Errors for {lidar_name}:")
    print(f"  Translation Errors [x, y, z]: {translation_errors}")
    print(f"  Translation Error (RMSE) [m]: {translation_error_rmse}")
    print(f"  Rotation Error (Degrees): {rotation_error_degrees}")
    print(f"  Rotation Errors Individual [r, p, y]: {rotation_errors_individual}")
    print("\n")

    return translation_error_rmse, rotation_error_degrees
    
