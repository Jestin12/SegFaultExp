import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jestin/SegFaultExp/major/major_ws/install/line_follower'
