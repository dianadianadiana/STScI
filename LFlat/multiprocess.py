from multiprocessing import Pool
import time

def worker(inputs):
    """
    Worker function
    """
    x, y = inputs
    return x / 2.


def other(num, num_cpu=4):
    """
    print num numbers
    """

    runs = []
    for i in range(num):
        runs.append((i, "hi: " + str(i)))

    pool = Pool(processes=num_cpu)
    results = pool.map_async(worker, runs)
    pool.close()
    pool.join()

    #print ""
    #print " Joined! "

    final = []
    for res in results.get():
        #print '***', res
        final.append(res)

    return final
start_time = time.time()
final = other(5,2)
print "%s seconds for processing" % (time.time() - start_time)
# .118 for num=50, num_cpu=1
# .124 for num=50, num_cpu=2
# .145 for num=50, num_cpu=5
# .177 for num=50, num_cpu=8

import numpy as np

def get_avgtime(func, itera, *args):
    times = np.array([])
    for i in range(itera):
        start_time = time.time()
        func(*args)
        times = np.append(times, time.time() - start_time)
    return np.mean(times) 
    
print get_avgtime(other, 100, 5,8)

#.159s = get_avgtime(other, 50, 5000,8)
#.127s = get_avgtime(other, 50, 5000,2)
#.131s = get_avgtime(other, 50, 5000,4)
#.118s = get_avgtime(other, 50, 5000,1)
#.157s = get_avgtime(other, 50, 5000,10)
#.202 = get_avgtime(other, 50, 5000,20)

#.114 = get_avgtime(other, 100, 5,2)
#.122 = get_avgtime(other, 100, 5,4)
#.128 = get_avgtime(other, 100, 5,6)
#.134 = get_avgtime(other, 100, 5,8)