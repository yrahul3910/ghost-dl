import os
import warnings
import numpy as np
import time 
import sys

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("../src/defect prediction/"))
sys.path.append(os.path.abspath("../src/"))
from main_d2h import _test, file_dic


times = []
for key in file_dic.keys():
    print(key)
    a = time.process_time()
    _test(key)
    b = time.process_time()
    print(b-a)
    times.append(b-a)

print(np.mean(times))
