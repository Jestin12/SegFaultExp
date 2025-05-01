import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/neel/Documents/MTRX5700/SegFaultExp/A3/pathfinder/install/pedestrian'
