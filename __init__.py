import os

rootPath = os.path.dirname(os.path.abspath(__file__))

HOME = os.path.expanduser('~')

rootDataPath = HOME + '/deepBND/DATA'
#rootDataPath = '/home/felipefr/switchdrive/scratch/deepBND'
#rootDataPath = '/Users/felipefr/switchdrive/scratch/deepBND'


__all__ = ['rootDataPath']
