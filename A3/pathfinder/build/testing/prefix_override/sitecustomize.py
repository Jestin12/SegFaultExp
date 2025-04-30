import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jestin/SegFaultExp/A3/pathfinder/install/testing'
