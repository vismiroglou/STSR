import numpy as np
import pandas as pd
import cv2
import yaml
import os

def get_coeffs(wtype: str, table_path:str, full=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieve absorption, scattering, attenuation, and diffuse attenuation coefficients for a given water type.

    Args:
        wtype (str): Water type identifier (e.g., 'I', 'IA', 'IB', 'II', 'III', '1C', '3C', '5C', '7C', '9C').
        table_path (str): Path to the CSV file containing water coefficients.   
        full (bool, optional): If True, returns coefficients for wavelengths 400–700nm (step 50nm).
                              If False, returns coefficients for 600nm, 550nm, and 450nm. Defaults to False.

    Returns:
        tuple:
            a_coeffs (np.ndarray): Absorption coefficients.
            b_coeffs (np.ndarray): Scattering coefficients.
            at_coeffs (np.ndarray): Attenuation coefficients (absorption + scattering).
            down_coeffs (np.ndarray): Diffuse attenuation coefficients.

    Raises:
        KeyError: If wtype is not in the mapping.
        FileNotFoundError: If 'jerlov.csv' is missing.
        ValueError: If expected columns are missing in the CSV.
    """
    mapping = {'I':0,
               'IA':1, 
               'IB':2,
               'II':3,
               'III':4,
               '1C':5,
               '3C':6,
               '5C':7,
               '7C':8,
               '9C':9}
    csv = pd.read_csv(table_path, dtype={"wtype": str})
    coeffs = csv.loc[csv['wtype'] == wtype].squeeze()

    if full:
        nm_range = range(400, 701, 50)
        a_coeffs = np.array([coeffs[f'a_{nm}'] for nm in nm_range])
        b_coeffs = np.array([coeffs[f'b_{nm}'] for nm in nm_range])
        at_coeffs = a_coeffs + b_coeffs
        down_coeffs = np.array([coeffs[f'K_{nm}'] for nm in nm_range])
    else:   
        a_coeffs = np.array([coeffs['a_600'], coeffs['a_550'], coeffs['a_450']])
        b_coeffs = np.array([coeffs['b_600'], coeffs['b_550'], coeffs['b_450']])
        at_coeffs = a_coeffs + b_coeffs
        down_coeffs = np.array([coeffs['K_600'], coeffs['K_550'], coeffs['K_450']])

    return a_coeffs, b_coeffs, at_coeffs, down_coeffs


def get_updated_coeffs(z:np.ndarray, coeffs:np.ndarray, sr:pd.DataFrame) -> np.ndarray:
    """
    Updates attenuation coefficients for each RGB channel based on depth and spectral response.

    Args:
        z (np.ndarray): 2D array of depth values (height x width).
        coeffs (array-like): Array of attenuation coefficients for each spectral band.
        sr (pandas.DataFrame): Spectral response DataFrame with columns for wavelength bands and a channel identifier.

    Returns:
        np.ndarray: 3D array (height x width x 3) of updated attenuation coefficients for R, G, B channels.

    Notes:
        - The function assumes spectral bands from 400nm to 700nm in steps of 50nm.
        - The spectral response DataFrame should have a column indicating channel ('R', 'G', 'B') and columns for each wavelength.
    """
    fullsize_coeffs = np.tile(coeffs, (z.shape[0], z.shape[1], 1))

    height, width = z.shape
    updated_coeffs = np.zeros((height, width, 3), dtype=np.float32)

    channels = ['R', 'G', 'B']

    for i, ch in enumerate(channels):
        frac_top = 0
        frac_bottom = np.zeros((height, width), dtype=np.float32)
        for nm in range(400, 701, 50):
            sc = sr[sr.iloc[:, 1] == ch][str(nm)].values[0]
            band_idx = int((nm - 400) / 50)
            attenuation = np.exp(-z * fullsize_coeffs[:, :, band_idx])
            frac_top += sc
            frac_bottom += (sc * attenuation)
        ch_coeff = np.log(np.full((height, width), frac_top) / frac_bottom) / z
        updated_coeffs[:, :, i] = ch_coeff
    return updated_coeffs


def get_sr(camera:str=None) -> tuple[pd.DataFrame, bool]:
    """
    Loads and processes camera-specific spectral response coefficients from a CSV file.

    If a camera name is provided, the function reads the camera specification database,
    selects the row corresponding to the given camera, splits metadata and spectral data,
    normalizes the spectral data so each row sums to 1, and combines the metadata and normalized data.
    Returns the processed DataFrame and a boolean indicating successful loading.

    Parameters:
        camera (str, optional): The name of the camera to load coefficients for. If None, no data is loaded.

    Returns:
        tuple[pd.DataFrame, bool]:
            - pd.DataFrame: The processed spectral response data for the specified camera, or None if no camera is provided.
            - bool: True if data was loaded and processed, False otherwise.
    """
    if camera:
        print('Loading camera specific coefficients for camera:', camera)
        cs = pd.read_csv('./data_tables/camspec_database.csv')
        sr = cs[cs.iloc[:, 0] == camera]
        # Split metadata and data
        metadata = sr.iloc[:, :2] 
        columns = list(str(num) for num in range(400, 701, 50))
        data = sr[columns] 
        # Normalize each row in 'data' to sum to 1
        row_sums = data.sum(axis=1)
        normalized_data = data.div(row_sums, axis=0)

        # Combine back
        sr = pd.concat([metadata, normalized_data], axis=1)
        full = True
    else:
        sr = None
        full = False
    return sr, full


def load_config(args: object) -> dict:
    """
    Loads configuration settings from CLI arguments and a YAML config file.

    CLI arguments take precedence over config file values, which in turn override defaults.

    Args:
        args (object): Namespace or object with attributes corresponding to config keys.

    Returns:
        dict: Dictionary containing the final configuration values.

    Raises:
        AssertionError: If 'input_img' is not provided via CLI or config.
    """
    def get(key: str, default=None):
        return getattr(args, key, None) or config.get(key, default)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f) or {}

    input_img = getattr(args, 'input_img', None) or config.get('input_img')
    assert input_img, "Error: 'input_img' must be provided via CLI or config."

    output_dir = getattr(args, 'output_dir', None) or config.get('output_dir')
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(input_img)), os.path.basename(input_img).split('.')[0])
    else:
        output_dir = os.path.join(output_dir, os.path.basename(input_img).split('.')[0])

    final_config = {
        'input_img': input_img,
        'output_dir': output_dir,
        'coeff_data_table': get('coeff_data_table', 'data_tables/randomized_water_types.csv'),
        'wtype': get('wtype', ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']),
        'vdepths': get('vdepths', [1]),
        'z_min': get('z_min', 0.1),
        'z_max': get('z_max', 5.0),
        'gamma': get('gamma', 1.0),
        'fscatter': get('fscatter', True),
        'g': get('g', 0.3),
        'mu': get('mu', 0.3),
        'inhomog': get('inhomog', True),
        'grf_min': get('grf_min', 0.5),
        'grf_max': get('grf_max', 1.5),
        'camera': get('camera', 'Nikon D90'),
    }

    return final_config


def load_img(img):
    if isinstance(img, str):
        img = cv2.imread(img)
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255).astype('float32')
    return img