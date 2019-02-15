import numpy as np, os, time, sys, random
from core import mod_neuro_evo as utils_ne
from core import mod_utils as utils
import gym, torch
from core import replay_memory
from core import ddpg as ddpg
import argparse
import time
import logging
import copy
import ray
import threading,queue
from ray.rllib.utils.timer import TimerStat
from torch.optim import Adam
import torch.nn as nn


render = False
parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) (Swimmer-v2) (Hopper-v2)', required=True)
env_tag = vars(parser.parse_args())['env']


logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(level=logging.DEBUG)


class Parameters:
    def __init__(self):
        #Number of Frames to Run
        if env_tag == 'Hopper-v2': self.num_frames = 4000000
        elif env_tag == 'Ant-v2': self.num_frames = 6000000
        elif env_tag == 'Walker2d-v2': self.num_frames = 8000000
        else: self.num_frames = 2000000

        #USE CUDA
        self.is_cuda = True; self.is_memory_cuda = True

        #Sunchronization Period
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.synch_period = 1
        else: self.synch_period = 10

        #DDPG params
        self.use_ln = True  # True
        self.gamma = 0.99; self.tau = 0.001
        self.seed = 7
        self.batch_size = 128
        self.buffer_size = 1000000
        self.frac_frames_train = 1.0
        self.use_done_mask = True

        ###### NeuroEvolution Params ########
        #Num of trials
        if env_tag == 'Hopper-v2' or env_tag == 'Reacher-v2': self.num_evals = 5
        elif env_tag == 'Walker2d-v2': self.num_evals = 3
        else: self.num_evals = 1

        #Elitism Rate
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.elite_fraction = 0.3
        elif env_tag == 'Reacher-v2' or env_tag == 'Walker2d-v2': self.elite_fraction = 0.2
        else: self.elite_fraction = 0.1

        self.pop_size = 10
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9

        #Save Results
        self.state_dim = None; self.action_dim = None #Simply instantiate them here, will be initialized later
        self.save_foldername = 'test3-debug/%s/' % env_tag
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

original = False


@ray.remote(num_gpus=0.1)
class Worker(object):
    def __init__(self, args,actor):
        # self.env = env_creator(config["env_config"]) # Initialize environment.
        # self.policy = ddpg.Actor(args)
        self.env = utils.NormalizedActions(gym.make(env_tag))
        self.args = args
        # self.args.is_cuda = True
        self.evolver = utils_ne.SSNE(self.args)
        # self.replay_buffer = replay_buff

        self.rl_agent = ddpg.DDPG(args, actor)
        # init rl agent
        # self.rl_agent = ddpg.DDPG(args)
        self.ounoise = ddpg.OUNoise(args.action_dim)
        # self.policy.eval()
        self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size//args.pop_size)
        # init ea pop
        # self.pop = dict([(key, ddpg.Actor(args))for key in range(args.pop_size)])
        # for i in range(args.pop_size):
        #     self.pop[i].eval()

        self.num_games = 0; self.num_frames = 0; self.gen_frames = 0
        # Details omitted.

    def set_gen_frames(self, value):
        self.gen_frames = value
        return self.gen_frames

    def get_gen_num(self):
        return self.gen_frames

    def copy_to_gpus(self):
        pass

    def epoch(self, all_fitness):
        return self.evolver.epoch(self.pop, all_fitness)

    def sample(self, batch):
        return self.replay_buffer.sample(batch)

    def ddpg_learning(self):
        if len(self.replay_buffer) > self.args.batch_size * 5:
            for _ in range(int(self.gen_frames * self.args.frac_frames_train)):
                sample_choose = np.random.randint(self.args.pop_size + 1)
                transitions_id = self.workers[sample_choose].sample.remote(self.args.batch_size)
                transitions = ray.get(transitions_id)
                # transitions = results_ea[sample_choose][0].sample(self.args.batch_size)
                batch = replay_memory.Transition(*zip(*transitions))
                self.rl_agent.update_parameters(batch)

            # self.gen_frames = 0
            return self.rl_agent.actor

    def add_experience(self, state, action, next_state, reward, done, replay_memroy):
        reward = utils.to_tensor(np.array([reward])).unsqueeze(0)
        if self.args.is_cuda: reward = reward.cuda()
        if self.args.use_done_mask:
            done = utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
            if self.args.is_cuda: done = done.cuda()
        action = utils.to_tensor(action)
        if self.args.is_cuda: action = action.cuda()
        replay_memroy.push(state, action, next_state, reward, done)

    def evaluate(self, model, num_evals, replay_memory, is_action_noise=False, store_transition=True):
        fitness = 0.0
        net = ddpg.Actor(self.args)
        net.load_state_dict(model)
        for _ in range(num_evals):
            fitness += self._evaluate(net, replay_memory, is_action_noise=is_action_noise, store_transition=store_transition)
        return fitness/num_evals, len(replay_memory), \
               self.num_frames, self.gen_frames, \
               self.num_games

    def _evaluate(self, net, replay_memory, is_render=False, is_action_noise=False, store_transition=True):
        total_reward = 0.0
        state = self.env.reset()
        state = utils.to_tensor(state).unsqueeze(0)
        if self.args.is_cuda:
            state = state.cuda()
        done = False

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            if render and is_render: self.env.render()
            # print(state)
            # exit(0)
            action = net.forward(state)
            action.clamp(-1, 1)
            action = utils.to_numpy(action.cpu())
            if is_action_noise: action += self.ounoise.noise()
            # print("come there in evaluate")
            next_state, reward, done, info = self.env.step(action.flatten())  # Simulate one step in environment
            # print("come there in evaluate")
            next_state = utils.to_tensor(next_state).unsqueeze(0)
            if self.args.is_cuda:
                next_state = next_state.cuda()
            total_reward += reward

            if store_transition: self.add_experience(state, action, next_state, reward, done,replay_memory)
            state = next_state
        if store_transition: self.num_games += 1
        # print("come here,total_reward:",total_reward)
        return total_reward


class Agent:
    def __init__(self, args, env):
        self.args = args; self.env = env
        self.evolver = utils_ne.SSNE(self.args)
        # self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size)
        self.pop = []
        for _ in range(args.pop_size):
            self.pop.append(ddpg.Actor(args))

        # for actor in self.pop: actor.eval()

        self.workers = [Worker.remote(args, actor) for actor in self.pop]

        # args.is_cuda = True; args.is_memory_cuda = True
        # self.rl_agent = ddpg.DDPG(args)
        # self.ounoise = ddpg.OUNoise(args.action_dim)
        # self.learner = LearnerThread()
        # self.learner.start()
        # self.learning_started = False

        # Stats
        self.timers = {
            k: TimerStat()
            for k in [
            "put_weights", "get_samples", "sample_processing",
            "replay_processing", "update_priorities", "train", "sample"
        ]
        }

        # self.num_games = 0; self.num_frames = 0; self.gen_frames = 0; self.len_replay = 0

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    def train(self):
        # self.gen_frames = 0
        # replay_buffer = replay_memory.ReplayMemory(self.args.buffer_size)
        # replay_buffer_id = ray.put(replay_buffer)

        print("begin training")
        # get_num_ids = [worker.set_gen_frames.remote(0) for worker in self.workers]
        # for worker in self.workers: worker.set_gen_frames.remote(0)

        ####################### EVOLUTION #####################
        # get_num_ids = [worker.get_gen_num.remote() for worker in self.workers]
        # gen_nums = ray.get(get_num_ids)

        ##### get new experiences
        # print("gen_nums:{0}".format(gen_nums))

        evaluate_timer = TimerStat()
        with evaluate_timer:
            evaluate_ids = [worker.evaluate.remote(self.pop[key].state_dict())
                            for key, worker in enumerate(self.workers)]
            results_ea = ray.get(evaluate_ids)
        print("evaluate_timer:{}".format(evaluate_timer.mean))


        ddpg_timer = TimerStat()
        with ddpg_timer:
            ddpg_ids = [worker.ddpg_learning.remote() for key, worker in enumerate(self.workers)]
            results_ddpg = ray.get(ddpg_ids)
        print("ddpg_timer:{}".format(ddpg_timer.mean))

        print(results_ddpg)

        exit(0)





        print("evaluate_timer:{}".format(evaluate_timer.mean))

        # logger.debug("results:{}".format(results_ea))

        all_fitness = []

        for i in range(self.args.pop_size):
            all_fitness.append(results_ea[i][0])

        logger.debug("fitness:{}".format(all_fitness))
        best_train_fitness = max(all_fitness)
        worst_index = all_fitness.index(min(all_fitness))

        #Validation test
        champ_index = all_fitness.index(max(all_fitness))
        logger.debug("champ_index:{}".format(champ_index))

        test_score_id = self.workers[0].evaluate.remote(self.pop[champ_index].state_dict(), 5, replay_buffer_id, store_transition=False)
        test_score = ray.get(test_score_id)[0]
        logger.debug("test_score:{0},champ_index:{1}".format(test_score, champ_index))

        # NeuroEvolution's probabilistic selection and recombination step
        elite_index = self.evolver.epoch(self.pop, all_fitness)
        ###################### DDPG #########################



        # result_rl_id = self.workers[-1].evaluate.remote(self.rl_agent.actor.state_dict(), 1 ,replay_buffer_id, is_action_noise=True) #Train
        # result_rl = ray.get(result_rl_id)
        #
        # logger.debug("results_rl:{}".format(result_rl))
        # results_ea.append(result_rl)

        # gen_frames = 0; num_games = 0; len_replay = 0;num_frames = 0
        sum_results = np.sum(results_ea, axis=0)
        # test = sum(results_ea)
        # fitness / num_evals, len(relay_buff), self.num_frames, self.gen_frames, self.num_games
        # logger.debug("test:{0},results_ea:{1}".format(sum_results, results_ea))
        #
        # self.len_replay = sum_results[1]
        # self.num_frames = sum_results[2]
        # self.gen_frames = sum_results[3]
        # self.num_games = sum_results[4]

        # test_timer = TimerStat()
        # print("gen_frames:{}".format(self.gen_frames))
        # print("test_timer:{}".format(test_timer.mean))

        return best_train_fitness, test_score, elite_index


class LearnerThread(threading.Thread):
    """Background thread that updates the local model from replay data.
    The learner thread communicates with the main thread through Queues. This
    is needed since Ray operations can only be run on the main thread. In
    addition, moving heavyweight gradient ops session runs off the main thread
    improves overall throughput.
    """
    def __init__(self, local_evaluator, ddpg):
        threading.Thread.__init__(self)
        self.learner_queue_size = WindowStat("size", 50)
        self.local_evaluator = local_evaluator
        self.inqueue = queue.Queue(maxsize=LEARNER_QUEUE_MAX_SIZE)
        self.outqueue = queue.Queue()
        # self.queue_timer = TimerStat()
        # self.grad_timer = TimerStat()
        self.daemon = True
        self.weights_updated = False
        self.stopped = False
        self.stats = {}

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        # with self.queue_timer:
        ra, replay = self.inqueue.get()
        if replay is not None:
            batch = replay.Transition(*zip(*transitions))
            prio_dict = {}
            # with self.grad_timer:
            grad_out = self.local_evaluator.compute_apply(replay)
            for pid, info in grad_out.items():
                prio_dict[pid] = (
                    replay.policy_batches[pid].data.get("batch_indexes"),
                    info.get("td_error"))
                if "stats" in info:
                    self.stats[pid] = info["stats"]
            self.outqueue.put((ra, prio_dict, replay.count))
        self.learner_queue_size.push(self.inqueue.qsize())
        self.weights_updated = True


if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = utils.Tracker(parameters, ['erl'], '_score.csv')  # Initiate tracker
    frame_tracker = utils.Tracker(parameters, ['frame_erl'], '_score.csv')  # Initiate tracker
    time_tracker = utils.Tracker(parameters, ['time_erl'], '_score.csv')

    # learner = LearnerThread(self.local_evaluator)
    # learner.start()

    #Create Env
    env = utils.NormalizedActions(gym.make(env_tag))
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    logger.debug("action_dim:{0},parameters.state_dim:{1}".format(parameters.action_dim,parameters.state_dim))

    #Seed
    env.seed(parameters.seed);
    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)

    #Create Agent
    ray.init(include_webui=False,ignore_reinit_error=True)
    # print(torch.cuda.device_count())

    agent = Agent(parameters, env)
    print('Running', env_tag, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)

    next_save = 100; time_start = time.time()
    while True#agent.num_frames <= parameters.num_frames:
        best_train_fitness, erl_score, elite_index = agent.train()
        print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:', '%.2f'%best_train_fitness if best_train_fitness != None else None, ' Test_Score:','%.2f'%erl_score if erl_score != None else None, ' Avg:','%.2f'%tracker.all_tracker[0][1], 'ENV '+env_tag)
        print('RL Selection Rate: Elite/Selected/Discarded', '%.2f'%(agent.evolver.selection_stats['elite']/agent.evolver.selection_stats['total']),
                                                             '%.2f' % (agent.evolver.selection_stats['selected'] / agent.evolver.selection_stats['total']),
                                                              '%.2f' % (agent.evolver.selection_stats['discarded'] / agent.evolver.selection_stats['total']))

        # log experiment result
        tracker.update([erl_score], agent.num_games)
        frame_tracker.update([erl_score], agent.num_frames)
        time_tracker.update([erl_score], time.time()-time_start)

        #Save Policy
        if agent.num_games > next_save:
            next_save += 100
            if elite_index != None: torch.save(agent.pop[elite_index].state_dict(), parameters.save_foldername + 'evo_net')
            print("Progress Saved")

        exit(0)











