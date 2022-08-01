import os

rootPath = os.path.dirname(os.path.abspath(__file__))

HOME = os.path.expanduser('~')

rootDataPath = HOME + '/deepBND/DATA'
# rootDataPath = HOME + '/switchdrive/scratch'
#rootDataPath = '/Users/felipefr/switchdrive/scratch/deepBND'


__all__ = ['rootDataPath']
