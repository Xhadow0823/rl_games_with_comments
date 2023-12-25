"""Runner 類別
"""
import os
import time
import numpy as np
import random
from copy import deepcopy
import torch

from rl_games.common import object_factory
from rl_games.common import tr_helpers

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent


def _restore(agent, args):
    "若 args 中有 checkpoint 則將其載入"
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        agent.restore(args['checkpoint'])

def _override_sigma(agent, args):
    "若 args 中有sigma 設定則覆蓋 agent.model.a2c_network 的 sigma 設定"
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Print cannot set new sigma because fixed_sigma is False')


class Runner:
    """用來建立 runner 實體\ 
    可以註冊要用的 algo 與 player 分別到 algo_factory 與 player_factory 之中，還有設定要用的 algo_observer。 \ 
    用 load() 輸入訓練設定。 \
    用 run() 來進行訓練 或 測試。 \
    可以用 create_player() 建立並獲得 player 實體。
    """
    
    def __init__(self, algo_observer=None):
        ''' 建立 runner 實體 \ 
        會先建立 algo factory 與 player factory 兩個實體， \ 
        並向這兩個實體註冊一些基本的 Agent 與 Player 的 constructor， \ 
        再建立 algo observer 。
        '''
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()
        torch.backends.cudnn.benchmark = True
        ### it didnot help for lots for openai gym envs anyway :(
        #torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)

    def reset(self):
        "no function now"
        pass

    def load_config(self, params):
        " user 用 load 就好，load 會再呼叫 load_config"
        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        if params["config"].get('multi_gpu', False):
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            # set different random seed for each GPU
            self.seed += self.global_rank

            print(f"global_rank = {self.global_rank} local_rank = {self.local_rank} world_size = {self.world_size}")

        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            # deal with environment specific seed if applicable
            if 'env_config' in params['config']:
                if not 'seed' in params['config']['env_config']:
                    params['config']['env_config']['seed'] = self.seed
                else:
                    if params["config"].get('multi_gpu', False):
                        params['config']['env_config']['seed'] += self

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_config):
        "會讀取 config dict 並載入到 .params 與 .default_config 中"
        config = deepcopy(yaml_config)
        self.default_config = deepcopy(config['params'])
        self.load_config(params=self.default_config)

    def run_train(self, args):
        """ user 通常會直接呼叫 .run 並傳入 args['train']==True \ 
        會利用 load 的 config 中的 algo_name 去 algo factory 中使用找到該 algo 的 constructor 並建立 algo 實體，命名為 agent。 \ 
        用 algo 實體 也就是 agent 呼叫 algo.train()。
        """
        print('Started to train')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)  # 在 factory 中用 algo_name 找到 XXXAgent 的 constructor，並建立實體（algo_name 之後的參數一併傳入建構）
        _restore(agent, args)
        _override_sigma(agent, args)
        agent.train()  # agent 會是任何 XXXAgent() 的實體

    def run_play(self, args):
        """ user 通常會直接呼叫 .run 並傳入 args['play']==True \ 
        會再呼叫 .create_player() 得到 player 實體。 \
        呼叫 player.run() 。
        """
        print('Started to play')
        player = self.create_player()  # 因為 user 通常會獨立拿出 player 來進行實驗，所以獨立出 create_player 介面 以得到 player 實體
        _restore(player, args)
        _override_sigma(player, args)
        player.run()  # player 會是 player.xxxPlayerxxx()的 實體

    def create_player(self):
        "會利用 algo_name 在 player_factory 中找到 player 的 constructor 並建立 player 實體。完成後回傳 player。"
        return self.player_factory.create(self.algo_name, params=self.params)

    def reset(self):
        "no function now"
        pass

    def run(self, args):
        "會檢查是 train 或是 play 模式，分別呼叫 run_train 和 run_play，輸入的 args 會一併傳下去。"
        if args['train']:
            self.run_train(args)
        elif args['play']:
            self.run_play(args)
        else:
            self.run_train(args)