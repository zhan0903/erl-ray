import ray


@ray.remote(num_gpus=1)
class Worker(object):
    def __init__(self, args):
        self.args = args
        self.gen_frames = 0

    def set_gen_frames(self, value):
        self.gen_frames = value
        return self.gen_frames

    def get_gen_num(self):
        return self.gen_frames


class Parameters:
    def __init__(self):
        self.is_cuda = False;
        self.is_memory_cuda = True
        self.pop_size = 10


if __name__ == "__main__":
    ray.init()
    args = Parameters()
    workers = [Worker.remote(args) for _ in range(args.pop_size)]
    # for worker in workers: worker.set_gen_frames.remote(0)
    get_num_ids = [worker.get_gen_num.remote() for worker in workers]
    gen_nums = ray.get(get_num_ids)
    print(gen_nums)
