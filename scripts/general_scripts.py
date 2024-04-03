import os


# Collection of functions for general eks scripting


def handle_io(csv_dir, save_dir):
    """ finds input and output directories """
    if not os.path.isdir(csv_dir):
        raise ValueError('--csv-dir must be a valid directory containing prediction csv files')
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(save_dir, exist_ok=True)
    return save_dir
