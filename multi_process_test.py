import multiprocessing

def print_hello(worker_id):
    print(f"Hello from worker {worker_id}")

if __name__ == '__main__':
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(print_hello, range(4))