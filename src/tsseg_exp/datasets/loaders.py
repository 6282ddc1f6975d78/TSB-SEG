import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path
import numpy as np
import pandas as pd
from tsseg.data.datasets import load_mocap
import re

def _get_all_params(dataset_name: str, data_root: Path) -> list[dict]:
    """
    Generates a list of all valid parameter combinations for a given dataset.

    This function scans the dataset's directory structure to find all
    available time series and their corresponding parameters.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    data_root : Path
        The path to the root data directory.

    Returns
    -------
    list[dict]
        A list of dictionaries, where each dictionary contains the parameters
        to load a single time series from the dataset.
    """
    params_list = []
    if dataset_name == 'pamap2':
        protocol_path = data_root / 'pamap2' / 'Protocol'
        if not protocol_path.exists(): return []
        for f in protocol_path.glob('subject10*.dat'):
            match = re.search(r'subject10(\d+)\.dat', f.name)
            if match:
                params_list.append({'subject_number': int(match.group(1))})
    elif dataset_name == 'usc-had':
        base_path = data_root / 'usc-had'
        if not base_path.exists(): return []
        subject_dirs = [d for d in base_path.glob('Subject*') if d.is_dir()]
        for s_dir in subject_dirs:
            s_num_match = re.search(r'Subject(\d+)', s_dir.name)
            if not s_num_match: continue
            s_num = int(s_num_match.group(1))
            
            # Find all unique target numbers for this subject
            target_numbers = set()
            for f in s_dir.glob('a*t*.mat'):
                t_num_match = re.search(r'a\d+t(\d+)\.mat', f.name)
                if t_num_match:
                    target_numbers.add(int(t_num_match.group(1)))
            
            for t_num in sorted(list(target_numbers)):
                # For USC-HAD, each trial is a concatenation of all activities
                # So we just need subject and target.
                params_list.append({'subject_number': s_num, 'target_number': t_num})
    
    elif dataset_name == 'actrectut':
        base_path = data_root / 'actrectut'
        if not base_path.exists(): return []
        for d in base_path.glob('subject*_walk'):
            match = re.search(r'subject(\d+)_walk', d.name)
            if match:
                params_list.append({'subject_number': int(match.group(1))})
    
    elif dataset_name in ['suturing', 'needle-passing', 'knot-tying']:
        base_path = data_root / dataset_name / 'transcriptions'
        if not base_path.exists(): return []
        for f in base_path.glob('*.txt'):
            subject, trial = f.stem.split('_', 1)
            params_list.append({'subject': subject, 'trial': trial})
    
    elif dataset_name == 'has':
        # Point to the preprocessed NPZ file.
        data_path = data_root / "has/has_data.npz"
        if not data_path.exists():
            # If the file doesn't exist, return an empty list.
            # The loader will then raise a helpful error.
            return []
        # The HAS dataset contains 250 time series with ts_id from 0 to 249.
        for i in range(250):
            params_list.append({'ts_id': i})

    elif dataset_name == 'pump':
        base_path = data_root / 'pump'
        if not base_path.exists():
            return []
        # Iterate over subdirectories like pump_v35, pump_v36...
        for version_dir in base_path.glob("pump_v*"):
            if version_dir.is_dir():
                 version_name = version_dir.name
                 for f in version_dir.glob("*.csv"):
                      params_list.append({'version': version_name, 'filename': f.name})

    # For tssb, utsa, skab, mocap, the logic is already handled in their loaders
    # or they don't fit this "all params" model well without more info.
    return params_list

def _load_pamap2(data_root: Path, subject_number: int):
    """Loads a trial from the PAMAP2 dataset."""
    ts_path = data_root / 'pamap2' / 'Protocol' / f'subject10{subject_number}.dat'
    df = pd.read_csv(ts_path, sep=' ', header=None)
    data = df.to_numpy()
    
    groundtruth = np.array(data[:, 1], dtype=int)
    # Columns: hand, chest, ankle accelerometer data
    hand_acc = data[:, 4:7]
    chest_acc = data[:, 21:24]
    ankle_acc = data[:, 38:41]
    X = np.hstack([hand_acc, chest_acc, ankle_acc])
    
    return X, groundtruth

def _load_usc_had(data_root: Path, subject_number: int, target_number: int):
    """
    Loads a single trial from the USC-HAD dataset, which consists of all
    activities concatenated for a given subject and target.
    """
    prefix = data_root / 'usc-had' / f'Subject{subject_number}'
    
    data_list = []
    label_list = []
    
    # Activities are numbered 1 to 12
    for activity_number in range(1, 13):
        fname = f'a{activity_number}t{target_number}.mat'
        file_path = prefix / fname
        if not file_path.exists():
            # Some subjects may not have all activity/target combinations
            continue
            
        mat_data = scipy.io.loadmat(file_path)
        activity_data = mat_data['sensor_readings']
        
        data_list.append(activity_data)
        label_list.append(np.full(len(activity_data), activity_number, dtype=int))
        
    if not data_list:
        raise FileNotFoundError(f"No data found for USC-HAD subject {subject_number}, target {target_number}")

    X = np.vstack(data_list)
    groundtruth = np.concatenate(label_list)
        
    return X, groundtruth

def _load_actrectut(data_root: Path, subject_number: int):
    """Loads a trial from the ActRecTut dataset."""
    ts_path = data_root / 'actrectut' / f'subject{subject_number}_walk' / 'data.mat'
    mat_data = scipy.io.loadmat(ts_path)
    groundtruth = mat_data['labels'].flatten()
    X = mat_data['data'][:, 0:10]
    return X, groundtruth

def _load_jigsaws(data_root: Path, dataset_name: str, subject: str, trial: str, variables: list[str] = None):
    """
    Loads a trial from a JIGSAWS dataset (suturing, knot-tying, needle-passing).
    """
    jigsaws_dataset_name_map = {
        'suturing': 'Suturing',
        'needle-passing': 'Needle_Passing',
        'knot-tying': 'Knot_Tying'
    }
    task_name = jigsaws_dataset_name_map.get(dataset_name)
    if not task_name:
        raise ValueError(f"Unknown JIGSAWS dataset name: {dataset_name}")

    file_stem = f"{task_name}_{subject}_{trial}"
    
    # --- Load groundtruth (gesture) data ---
    gt_path = data_root / dataset_name / 'transcriptions' / f"{subject}_{trial}.txt"
    with open(gt_path, 'r') as file:
        lines = file.read().strip().split('\n')
    
    if not lines or not lines[0]:
        raise ValueError(f"Empty groundtruth file: {gt_path}")

    max_index = max(int(line.split()[1]) for line in lines)
    gt_full = np.zeros(max_index + 1, dtype=int)

    for line in lines:
        start, end, gesture = line.split()
        start, end = int(start), int(end)
        gesture_num = int(re.findall(r'\d+', gesture)[0])
        gt_full[start:end+1] = gesture_num

    # --- Load time series data ---
    ts_path = data_root / dataset_name / 'kinematics' / 'AllGestures' / f"{subject}_{trial}.txt"
    df = pd.read_csv(ts_path, sep=r'\s+', header=None)

    if variables:
        variable_ranges = {
            'master_left_tooltip_xyz': list(range(0, 3)), 'master_left_tooltip_r': list(range(3, 12)),
            'master_left_tooltip_trans_vel': list(range(12, 15)), 'master_left_tooltip_rot_vel': list(range(15, 18)),
            'master_left_gripper_angle': [18], 'master_right_tooltip_xyz': list(range(19, 22)),
            'master_right_tooltip_r': list(range(22, 31)), 'master_right_tooltip_trans_vel': list(range(31, 34)),
            'master_right_tooltip_rot_vel': list(range(34, 37)), 'master_right_gripper_angle': [37],
            'slave_left_tooltip_xyz': list(range(38, 41)), 'slave_left_tooltip_r': list(range(41, 50)),
            'slave_left_tooltip_trans_vel': list(range(50, 53)), 'slave_left_tooltip_rot_vel': list(range(53, 56)),
            'slave_left_gripper_angle': [56], 'slave_right_tooltip_xyz': list(range(57, 60)),
            'slave_right_tooltip_r': list(range(60, 69)), 'slave_right_tooltip_trans_vel': list(range(69, 72)),
            'slave_right_tooltip_rot_vel': list(range(72, 75)), 'slave_right_gripper_angle': [75]
        }
        columns = [col for var in variables for col in variable_ranges.get(var, [])]
        ts_full = df.iloc[:, columns].to_numpy()
    else:
        ts_full = df.to_numpy()
        
    # --- Ensure consistency between labels and time series length ---
    # The groundtruth array might be longer than the actual time series if the
    # transcription file has an end index that is out of bounds.
    # We truncate the groundtruth to match the loaded time series length.
    if len(gt_full) > len(ts_full):
        gt_full = gt_full[:len(ts_full)]

    # --- Trim data to annotated region ---
    try:
        first_annotated_time = np.nonzero(gt_full)[0][0]
        last_annotated_time = np.nonzero(gt_full)[0][-1]
    except IndexError: # Handle cases with no annotations
        return np.array([]).reshape(0, ts_full.shape[1]), np.array([])

    gt = gt_full[first_annotated_time:last_annotated_time+1]
    ts = ts_full[first_annotated_time:last_annotated_time+1]
    
    return ts, gt

def _load_tssb(data_root: Path, ts_name: str = None):
    """
    Loads time series from the TSSB dataset.
    If ts_name is provided, loads a single time series.
    If ts_name is None, loads all time series from the dataset.
    """
    tssb_path = data_root / "tssb"
    desc_filename = tssb_path / "desc.txt"
    prop_filename = tssb_path / "properties.txt"

    if not desc_filename.exists() or not prop_filename.exists():
        # If metadata files are missing, return empty lists or raise a more specific error
        # depending on whether we are trying to load all or one.
        if ts_name:
             raise FileNotFoundError(f"Metadata files not found in {tssb_path}")
        return [], []

    with open(desc_filename, 'r') as f:
        all_ts_info = [line.strip().split(',') for line in f]

    if ts_name:
        ts_info_list = [info for info in all_ts_info if info[0] == ts_name]
        if not ts_info_list:
            raise ValueError(f"Time series '{ts_name}' not found in TSSB dataset.")
    else:
        ts_info_list = all_ts_info

    all_X = []
    all_y = []
    for ts_info in ts_info_list:
        current_ts_name = ts_info[0]
        change_points = [int(cp) for cp in ts_info[2:] if cp]

        # Find labels from properties.txt
        labels = []
        with open(prop_filename, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if parts[0] == current_ts_name:
                    label_cut = int(parts[2])
                    labels = [int(l) // (label_cut + 1) for l in parts[4:] if l]
                    break

        # Load time series data
        ts_filepath = tssb_path / f"{current_ts_name}.txt"
        X = np.loadtxt(fname=ts_filepath, dtype=np.float64)

        # Create groundtruth from change points and labels
        groundtruth = np.zeros(len(X), dtype=int)
        start_idx = 0
        for i, cp in enumerate(change_points):
            groundtruth[start_idx:cp] = labels[i]
            start_idx = cp
        groundtruth[start_idx:] = labels[-1]
        
        # Ensure X is 2D (n_timepoints, n_channels)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if ts_name: # If a specific ts_name was requested, return single X, y
            return X, groundtruth

        all_X.append(X)
        all_y.append(groundtruth)

    return all_X, all_y

def _load_utsa(data_root: Path, ts_name: str = None):
    """
    Loads time series from the UTSA dataset.
    If ts_name is provided, loads a single time series.
    If ts_name is None, loads all time series from the dataset.
    """
    utsa_path = data_root / "utsa"
    desc_filename = utsa_path / "desc.txt"
    prop_filename = utsa_path / "properties.txt"

    if not desc_filename.exists() or not prop_filename.exists():
        if ts_name:
             raise FileNotFoundError(f"Metadata files not found in {utsa_path}")
        return [], []

    with open(desc_filename, 'r') as f:
        all_desc_rows = [line.strip().split(',') for line in f]
    
    with open(prop_filename, 'r') as f:
        all_prop_rows = [line.strip().split(',') for line in f]

    if len(all_desc_rows) != len(all_prop_rows):
        raise ValueError("Description and property files have a different number of records in UTSA dataset.")

    if ts_name:
        try:
            # Find the specific time series info
            desc_row = next(row for row in all_desc_rows if row[0] == ts_name)
            prop_row = next(row for row in all_prop_rows if row[0] == ts_name)
        except StopIteration:
            raise ValueError(f"Time series '{ts_name}' not found in UTSA dataset.")
        
        ts_info_list = [(desc_row, prop_row)]
    else:
        ts_info_list = zip(all_desc_rows, all_prop_rows)

    all_X = []
    all_y = []

    for desc_row, prop_row in ts_info_list:
        if desc_row[0] != prop_row[0]:
            raise ValueError(f"Mismatched records for '{desc_row[0]}' in UTSA dataset.")
            
        current_ts_name = desc_row[0]
        # desc_row[1] is window_size, which is not used here.
        change_points = [int(cp) for cp in desc_row[2:] if cp]
        labels = [int(l) for l in prop_row[1:] if l]

        # Load time series data
        ts_filepath_txt = utsa_path / f"{current_ts_name}.txt"
        ts_filepath_npz = utsa_path / "data.npz"
        
        if ts_filepath_txt.exists():
            X = np.loadtxt(fname=ts_filepath_txt, dtype=np.float64)
        elif ts_filepath_npz.exists():
            with np.load(file=ts_filepath_npz) as data:
                if current_ts_name in data:
                    X = data[current_ts_name]
                else:
                    raise FileNotFoundError(f"Time series '{current_ts_name}' not found in data.npz for UTSA dataset.")
        else:
            raise FileNotFoundError(f"Data file for '{current_ts_name}' not found in UTSA dataset.")

        # Create groundtruth from change points and labels
        groundtruth = np.zeros(len(X), dtype=int)
        start_idx = 0
        for i, cp in enumerate(change_points):
            groundtruth[start_idx:cp] = labels[i]
            start_idx = cp
        groundtruth[start_idx:] = labels[-1]
        
        # Ensure X is 2D (n_timepoints, n_channels)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if ts_name: # If a specific ts_name was requested, return single X, y
            return X, groundtruth

        all_X.append(X)
        all_y.append(groundtruth)

    return all_X, all_y

def _load_skab(data_root: Path, ts_name: str = None):
    """
    Loads time series from the SKAB dataset.
    If ts_name is provided, loads a single time series.
    If ts_name is None, loads all time series from the dataset.
    """
    skab_path = data_root / "skab"
    desc_filename = skab_path / "desc.txt"
    prop_filename = skab_path / "properties.txt"

    if not desc_filename.exists() or not prop_filename.exists():
        if ts_name:
             raise FileNotFoundError(f"Metadata files not found in {skab_path}")
        return [], []

    with open(desc_filename, 'r') as f:
        all_desc_rows = [line.strip().split(',') for line in f]
    
    with open(prop_filename, 'r') as f:
        all_prop_rows = [line.strip().split(',') for line in f]

    if len(all_desc_rows) != len(all_prop_rows):
        raise ValueError("Description and property files have a different number of records in SKAB dataset.")

    if ts_name:
        try:
            # Find the specific time series info
            desc_row = next(row for row in all_desc_rows if row[0] == ts_name)
            prop_row = next(row for row in all_prop_rows if row[0] == ts_name)
        except StopIteration:
            raise ValueError(f"Time series '{ts_name}' not found in SKAB dataset.")
        
        ts_info_list = [(desc_row, prop_row)]
    else:
        ts_info_list = zip(all_desc_rows, all_prop_rows)

    all_X = []
    all_y = []

    for desc_row, prop_row in ts_info_list:
        if desc_row[0] != prop_row[0]:
            raise ValueError(f"Mismatched records for '{desc_row[0]}' in SKAB dataset.")
            
        current_ts_name = desc_row[0]
        change_points = [int(cp) for cp in desc_row[2:] if cp]
        labels = [int(l) for l in prop_row[1:] if l]

        # Load time series data from data.npz
        ts_filepath_npz = skab_path / "data.npz"
        if not ts_filepath_npz.exists():
            raise FileNotFoundError(f"data.npz not found for SKAB dataset.")
        
        with np.load(file=ts_filepath_npz) as data:
            if current_ts_name in data:
                X = data[current_ts_name]
            else:
                raise FileNotFoundError(f"Time series '{current_ts_name}' not found in data.npz for SKAB dataset.")

        # Create groundtruth from change points and labels
        groundtruth = np.zeros(len(X), dtype=int)
        start_idx = 0
        for i, cp in enumerate(change_points):
            groundtruth[start_idx:cp] = labels[i]
            start_idx = cp
        groundtruth[start_idx:] = labels[-1]
        
        # Ensure X is 2D (n_timepoints, n_channels)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if ts_name: # If a specific ts_name was requested, return single X, y
            return X, groundtruth

        all_X.append(X)
        all_y.append(groundtruth)

    return all_X, all_y

def _load_has(data_root: Path, ts_id: int):
    """Loads a time series from the preprocessed HAS dataset (.npz) by its ID."""
    # Correctly point to the preprocessed .npz file inside the 'has' subdirectory
    data_path = data_root / "has/has_data.npz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. "
            "Please run the 'scripts/preprocess_has.py' script once to generate it."
        )

    with np.load(data_path) as data:
        x_key = f"X_{ts_id}"
        y_key = f"y_{ts_id}"
        
        if x_key not in data or y_key not in data:
            raise ValueError(f"Time series with ts_id {ts_id} not found in {data_path}.")
            
        X = data[x_key]
        groundtruth = data[y_key]

    return X, groundtruth

# def _load_has(data_root: Path, ts_id: int):
#     """Loads a time series from the HAS dataset by its ID."""
#     data_path = data_root / "has2023_master.csv"

#     np_cols = ["change_points", "activities", "x-acc", "y-acc", "z-acc",
#                "x-gyro", "y-gyro", "z-gyro",
#                "x-mag", "y-mag", "z-mag",
#                "lat", "lon", "speed"]
#     converters = {col: lambda val: np.array(eval(val)) if val else np.array([]) for col in np_cols}

#     # Read only the required row from the CSV
#     df_iter = pd.read_csv(data_path, converters=converters, iterator=True, chunksize=100)
#     row = None
#     for chunk in df_iter:
#         if ts_id in chunk['ts_challenge_id'].values:
#             row = chunk[chunk['ts_challenge_id'] == ts_id].iloc[0]
#             break
    
#     if row is None:
#         raise ValueError(f"Time series with ts_challenge_id {ts_id} not found in HAS dataset.")

#     # Create groundtruth labels
#     label_mapping = {label: idx for idx, label in enumerate(np.unique(row.activities))}
#     groundtruth = np.array([label_mapping[label] for label in row.activities])

#     # Stack sensor data to form X
#     if row.group == "indoor":
#         ts_list = [row[f"{axis}-{sensor}"].reshape(-1, 1) for sensor in ["acc", "gyro", "mag"] for axis in "xyz"]
#     elif row.group == "outdoor":
#         ts_list = [row[f"{axis}-{sensor}"].reshape(-1, 1) for sensor in ["acc", "mag"] for axis in "xyz"]
#     else:
#         raise ValueError(f"Unknown group '{row.group}' in HAS dataset.")
    
#     X = np.hstack(ts_list)

#     return X, groundtruth

def _load_pump(data_root: Path, version: str, filename: str):
    """
    Loads a single trial from the Pump dataset.
    
    Parameters
    ----------
    data_root : Path
        Root data directory.
    version : str
        The version folder name (e.g. 'pump_v35').
    filename : str
        The CSV filename (e.g. 'Pump_A_DC35_0.csv').
        
    Returns
    -------
    X : np.ndarray
        Array of sensor readings (Sensor 1..9).
    y : np.ndarray
        Array of state labels.
    """
    file_path = data_root / 'pump' / version / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Pump file not found: {file_path}")
        
    # Read CSV. Header is presumed to exist.
    df = pd.read_csv(file_path)
    
    # Last column is 'state', others are sensors
    # Check if 'state' is in columns, else assume last column
    if 'state' in df.columns:
        y = df['state'].to_numpy(dtype=int)
        X = df.drop(columns=['state']).to_numpy()
    else:
        # Fallback if header is missing or different
        data = df.to_numpy()
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        
    return X, y
def load_dataset(dataset_name: str, data_root: str = "data/", return_X_y=True, **params):
    """
    Loads time series data from a specified dataset.

    This function acts as a dispatcher. If specific parameters (e.g.,
    `subject_number`, `ts_name`) are provided, it loads a single time series.
    If no parameters are given, it attempts to load all available time series
    from the dataset, returning them as lists of arrays.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load (e.g., 'pamap2', 'suturing', 'tssb').
    data_root : str, default="data/"
        The path to the root directory containing all dataset folders.
    return_X_y : bool, default=True
        If True, returns (X, y) tuple.
    **params : dict
        Keyword arguments specific to the dataset loader. If empty, the function
        will try to load all time series from the dataset.
        - 'pamap2': `subject_number` (int)
        - 'usc-had': `subject_number` (int), `target_number` (int)
        - 'actrectut': `subject_number` (int)
        - JIGSAWS ('suturing', etc.): `subject` (str), `trial` (str)
        - 'tssb', 'utsa', 'skab': `ts_name` (str, optional)
        - 'has': `ts_id` (int)
        - 'mocap': `trial` (int or str, optional)

    Returns
    -------
    X : np.ndarray or list of np.ndarray
    y : np.ndarray or list of np.ndarray
        Ground truth labels. A list is returned if all series are loaded.
    """
    data_root_path = Path(data_root)
    
    # If no specific parameters are passed, try to load all time series
    if not params:
        # Datasets with built-in "load all" logic
        # Datasets with built-in "load all" logic
        if dataset_name in ['tssb', 'utsa', 'skab']:
            if dataset_name == 'tssb':
                X, y = _load_tssb(data_root_path)
            elif dataset_name == 'utsa':
                X, y = _load_utsa(data_root_path)
            elif dataset_name == 'skab':
                X, y = _load_skab(data_root_path)
        elif dataset_name == 'mocap':
            # The load_mocap function can't load all by itself.
            # We assume we can iterate through trials by index.
            # This is a bit of a guess without seeing MOCAP_TRIALS, but it's a common pattern.
            all_X, all_y = [], []
            trial_idx = 0
            while True:
                try:
                    # Recursively call this function with a specific trial index
                    single_X, single_y = load_dataset(dataset_name, data_root, return_X_y=True, trial=trial_idx)
                    all_X.append(single_X)
                    all_y.append(single_y)
                    trial_idx += 1
                except ValueError as e:
                    # Assuming a ValueError on an invalid trial index means we're done
                    if "Invalid trial index" in str(e):
                        break
                    else:
                        raise e # Re-raise other ValueErrors
            if not all_X:
                 raise ValueError("Could not load any trials for mocap dataset.")
            X, y = all_X, all_y
        else:
            # Use the helper to find all parameter combinations
            all_params = _get_all_params(dataset_name, data_root_path)
            if not all_params:
                raise ValueError(f"Could not find any time series for dataset '{dataset_name}'. "
                                 "Please provide specific parameters to load a single series.")
            
            all_X, all_y = [], []
            for p in all_params:
                # Recursively call this function with specific params for one series
                single_X, single_y = load_dataset(dataset_name, data_root, return_X_y=True, **p)
                all_X.append(single_X)
                all_y.append(single_y)
            X, y = all_X, all_y
    else:
        # Load a single time series using provided parameters
        if dataset_name == 'pamap2':
            X, y = _load_pamap2(data_root_path, **params)
        elif dataset_name == 'usc-had':
            X, y = _load_usc_had(data_root_path, **params)
        elif dataset_name == 'actrectut':
            X, y = _load_actrectut(data_root_path, **params)
        elif dataset_name in ['suturing', 'needle-passing', 'knot-tying']:
            X, y = _load_jigsaws(data_root_path, dataset_name, **params)
        elif dataset_name == 'tssb':
            X, y = _load_tssb(data_root_path, **params)
        elif dataset_name == 'utsa':
            X, y = _load_utsa(data_root_path, **params)
        elif dataset_name == 'skab':
            X, y = _load_skab(data_root_path, **params)
        elif dataset_name == 'has':
            X, y = _load_has(data_root_path, **params)
        elif dataset_name == 'pump':
            X, y = _load_pump(data_root_path, **params)
        elif dataset_name == 'mocap':
            # The load_mocap function from tsseg does not use data_root
            X, y = load_mocap(**params, return_X_y=True)
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported by this loader.")
        
    if return_X_y:
        return X, y
    else:
        # This part might need adjustment if X and y are lists
        if isinstance(X, list):
            # Cannot easily convert list of arrays with different lengths to DataFrame
            raise ValueError("Cannot return as DataFrame when multiple series are loaded.")
        return pd.DataFrame(X), pd.Series(y)