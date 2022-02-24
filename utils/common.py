import multiprocessing


def get_cpu_count(usage_ratio):
    return round(multiprocessing.cpu_count() * usage_ratio)
