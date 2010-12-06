import cv
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

def cached_times(n):
    def decorator(func):
        memory = dict(
            call_counter = 0,
            cache = None
        )
        def cached_func(*args, **kwargs):
            if memory['call_counter'] % n == 0:
                memory['cache'] = func(*args, **kwargs)
            memory['call_counter'] = (memory['call_counter'] + 1) % n
            return memory['cache']
        return cached_func
    return decorator


def compute_time(func):
    def timed_func(*args, **kwargs):
        t = cv.GetTickCount()
        ret = func(*args, **kwargs)
        t = cv.GetTickCount() - t
        logging.debug("%s time = %gms" % (func.func_name, t / (cv.GetTickFrequency() * 1000.)))
        return ret
    return timed_func
