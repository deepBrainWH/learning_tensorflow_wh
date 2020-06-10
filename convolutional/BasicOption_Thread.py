# -*- coding: utf-8 -*-
'''
    python file describe:multithreading
'''
import numpy as np
import threading
import time
import tensorflow as tf

def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand()<0.1:
            print("stoping from id: %d" % worker_id)
            coord.request_stop()
        else:
            print("working on id:%d" % worker_id)
        time.sleep(1)

# 创建，启动，并推出线程
coord = tf.train.Coordinator()
threads = [threading.Thread(target=MyLoop, args=(coord, i)) for i in range(5)]
for t in threads:
    t.start()
coord.join(threads)
