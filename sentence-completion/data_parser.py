import config
import sys
import glob

sys.path.append(config.TRAINING_DATA_PATH)
def get_file_names(directory=config.TRAINING_DATA_PATH):
    return list(glob.glob('*.txt'))

print(get_file_names())
