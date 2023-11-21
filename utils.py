from pathlib import Path
import pickle


def pickle_bobj(bobj, save_filedir):
    """bobj means binary object"""
    if Path(save_filedir).exists():
        raise FileExistsError
    with open(save_filedir, 'wb') as f:
        pickle.dump(bobj, f)


def load_bobj(save_filedir):
    with open(save_filedir, 'rb') as f:
        bobj = pickle.load(f)
    return bobj
