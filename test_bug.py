import ray
import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, args, init=False):
        super(Actor, self).__init__()
        # TorchModel.__init__(self,args,init=False)
        self.args = args
        l1 = 128; l2 = 128; l3 = l2

        # Construct Hidden Layer 1
        self.w_l1 = nn.Linear(args.state_dim, l1)
        if self.args.use_ln: self.lnorm1 = LayerNorm(l1)

        #Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)
        self.w_l2 = nn.DataParallel(self.w_l2)
        if self.args.use_ln: self.lnorm2 = LayerNorm(l2)

        #Out
        self.w_out = nn.Linear(l3, args.action_dim)

        #Init
        if init:
            self.w_out.weight.data.mul_(0.1)
            self.w_out.bias.data.mul_(0.1)

        if args.is_cuda: self.cuda()

    def forward(self, input):
        #Hidden Layer 1
        out = self.w_l1(input)
        if self.args.use_ln: out = self.lnorm1(out)
        out = torch.tanh(out)

        #Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = torch.tanh(out)

        #Out
        out = torch.tanh(self.w_out(out))
        return out

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class OUNoise:
    def __init__(self, action_dimension, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


@ray.remote(num_gpus=0.1)
class Worker(object):
    def __init__(self, args):
        self.args = args
        self.gen_frames = 0

    def set_gen_frames(self, value):
        self.gen_frames = value
        return self.gen_frames

    def test(self,input):
        print("come here")

    def get_gen_num(self):
        return self.gen_frames


class Parameters:
    def __init__(self):
        self.is_cuda = False;
        self.is_memory_cuda = True
        self.pop_size = 10
        self.state_dim = 8
        self.action_dim = 2


if __name__ == "__main__":
    ray.init()
    args = Parameters()
    pop = []
    for _ in range(args.pop_size):
        pop.append(Actor(args))

    workers = [Worker.remote(args) for _ in range(args.pop_size)]
    get_num_ids = [worker.get_gen_num.remote() for worker in workers]
    gen_nums = ray.get(get_num_ids)
    print(gen_nums)

    evaluate_ids = [worker.test.remote(pop[key].state_dict(), args.num_evals)
                    for key, worker in enumerate(workers)]

    results_ea = ray.get(evaluate_ids)
    print("results:{}".format(results_ea))
