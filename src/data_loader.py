import os
import urllib.request
import pandas as pd
import numpy as np
from scipy.io import arff
from src.utils import logger, RANDOM_STATE

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def load_uci_seizure() -> tuple[np.ndarray, np.ndarray]:
    target_dir = os.path.join(DATA_DIR, 'uci_seizure')
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, 'data.csv')
    
    if not os.path.exists(file_path):
        logger.info("Downloading UCI Epileptic Seizure Recognition dataset...")
        # Primary: GitHub mirror (UCI direct URL changed after 2023 repo migration)
        url = "https://raw.githubusercontent.com/cyrille-feu/Logistic-Regression-on-Epileptic-Seizure-Recognition-Dataset/master/data.csv"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as response, open(file_path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            logger.warning(f"Failed to download UCI Seizure dataset: {e}. Returning synthetic data.")
            np.random.seed(RANDOM_STATE)
            return np.random.randn(11500, 178), np.random.randint(0, 2, 11500)
            
    df = pd.read_csv(file_path)
    # Binary: y=1 -> 1, y=2..5 -> 0
    y_raw = df['y'].values
    y = (y_raw == 1).astype(int)
    # Features: drop unnamed and y
    X = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed') or col == 'y']).values
    return X, y

def _generate_synthetic_bonn():
    logger.warning("Generating synthetic Bonn-like data for development...")
    np.random.seed(RANDOM_STATE)
    X = np.random.randn(500, 4097)
    y = np.zeros(500, dtype=int)
    y[:200] = 1  # 200 positive, 300 negative
    return X, y

def _extract_bonn_features(X_raw: np.ndarray) -> np.ndarray:
    """Extract statistical features from raw (500, 4097) signal."""
    from scipy.stats import skew, kurtosis
    features = []
    for row in X_raw:
        m = np.mean(row)
        s = np.std(row)
        sk = skew(row)
        k = kurtosis(row)
        e = np.sum(row**2)
        zcr = np.sum(np.diff(row > 0)) / len(row)
        features.append([m, s, sk, k, e, zcr])
    return np.array(features)

def load_bonn_eeg(raw: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Bonn EEG dataset.

    raw=False (default): returns (500, 6) statistical features — mean, std, skew,
        kurtosis, energy, zero-crossing rate. Fast. Used by Pipelines A and B.
    raw=True: returns (500, 4097) raw signals. Used by Pipeline C (WaveletDenoiser)
        which requires temporally meaningful sequential data.

    This dual-representation design is intentional:
    - Pipelines A/B operate on compact statistical features (6 dims) — fast and valid.
    - Pipeline C operates on raw signals (4097 dims) — required for wavelet denoising.
    """
    target_dir = os.path.join(DATA_DIR, 'bonn_eeg')
    os.makedirs(target_dir, exist_ok=True)

    sets = ['Z', 'O', 'N', 'F', 'S']
    has_all = all(os.path.exists(os.path.join(target_dir, s)) for s in sets)

    if not has_all:
        logger.info("Bonn dataset files not found. Attempting download from ukbonn.de...")
        bonn_zips = {
            'Z': 'https://www.ukbonn.de/site/assets/files/21874/z.zip',
            'O': 'https://www.ukbonn.de/site/assets/files/21872/o.zip',
            'N': 'https://www.ukbonn.de/site/assets/files/21871/n.zip',
            'F': 'https://www.ukbonn.de/site/assets/files/21870/f.zip',
            'S': 'https://www.ukbonn.de/site/assets/files/21875/s.zip',
        }
        try:
            import zipfile, io, shutil
            for s, url in bonn_zips.items():
                s_dir = os.path.join(target_dir, s)
                os.makedirs(s_dir, exist_ok=True)
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    z.extractall(s_dir)
                inner = os.path.join(s_dir, s)
                if os.path.isdir(inner):
                    for f in os.listdir(inner):
                        shutil.move(os.path.join(inner, f), os.path.join(s_dir, f))
                    os.rmdir(inner)
                mac = os.path.join(s_dir, '__MACOSX')
                if os.path.isdir(mac):
                    shutil.rmtree(mac)
                logger.info(f"  SET {s}: downloaded OK")
            has_all = True
        except Exception as e:
            logger.warning(f"Failed to download Bonn EEG: {e}. Using synthetic fallback.")
            X_raw = np.random.randn(500, 4097)
            y = np.zeros(500, dtype=int); y[:200] = 1
            return (X_raw, y) if raw else (_extract_bonn_features(X_raw), y)

    if not has_all:
        logger.warning("Bonn dataset not found. Using synthetic fallback.")
        X_raw = np.random.randn(500, 4097)
        y = np.zeros(500, dtype=int); y[:200] = 1
        return (X_raw, y) if raw else (_extract_bonn_features(X_raw), y)

    logger.info(f"Loading Bonn EEG ({'raw 4097-dim signals' if raw else 'extracted 6-dim features'})...")
    X_raw_list, y_list = [], []
    label_map = {'F': 1, 'S': 1, 'Z': 0, 'O': 0, 'N': 0}

    for s in sets:
        s_dir = os.path.join(target_dir, s)
        for fname in sorted(os.listdir(s_dir)):
            if fname.lower().endswith('.txt'):
                filepath = os.path.join(s_dir, fname)
                with open(filepath, 'r') as f:
                    vals = [float(line.strip()) for line in f if line.strip()]
                X_raw_list.append(vals)
                y_list.append(label_map[s])

    X_raw = np.array(X_raw_list)
    y = np.array(y_list)
    return (X_raw, y) if raw else (_extract_bonn_features(X_raw), y)

def load_eeg_eye_state() -> tuple[np.ndarray, np.ndarray]:
    target_dir = os.path.join(DATA_DIR, 'eeg_eye_state')
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, 'EEG_Eye_State.arff')
    
    if not os.path.exists(file_path):
        logger.info("Downloading UCI EEG Eye State dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            logger.warning(f"Failed to download EEG Eye State dataset: {e}. Returning synthetic data.")
            np.random.seed(RANDOM_STATE)
            return np.random.randn(14980, 14), np.random.randint(0, 2, 14980)
            
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    y_raw = df['eyeDetection'].values
    # ARFF byte string fix: scipy arff may return bytes on some platforms
    y = np.array([int(v.decode('utf-8')) if isinstance(v, bytes) else int(v) for v in y_raw])
    
    X = df.drop(columns=['eyeDetection']).values
    return X, y

def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == 'uci_seizure':
        return load_uci_seizure()
    elif name == 'bonn_eeg':
        return load_bonn_eeg(raw=False)   # 6 statistical features for Pipelines A/B
    elif name == 'bonn_eeg_raw':
        return load_bonn_eeg(raw=True)    # 4097 raw signals for Pipeline C only
    elif name == 'eeg_eye_state':
        return load_eeg_eye_state()
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_all_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns datasets keyed by name. Bonn is returned as 6-dim features (fast).
    Pipeline C uses load_bonn_eeg(raw=True) separately."""
    return {
        'uci_seizure':   load_uci_seizure(),
        'bonn_eeg':      load_bonn_eeg(raw=False),
        'eeg_eye_state': load_eeg_eye_state(),
    }
