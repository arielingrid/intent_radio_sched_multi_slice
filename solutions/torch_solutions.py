import gin
import numpy as np
import torch
import torch.nn as nn
#import torchvision.transforms as transforms
from solutions.base_solution import BaseSolution
from solutions.torch_modules import SelfAttentionMatrix
from solutions.torch_modules import AttentionNeuronLayer



torch.set_num_threads(1)


class BaseTorchSolution(BaseSolution):
    """Basic torch solution."""

    def __init__(self, device):
        self.modules_to_learn = []
        self.device = torch.device(device)

    def get_action(self, obs):
        with torch.no_grad():
            #print("get_action em base torch:",self._get_action(obs) )
            output = np.argmax(self._get_action(obs))
            #return output          ###Edited to output discrete values 
            return self._get_action(obs)

    def get_params(self):
        params = []
        with torch.no_grad():
            for layer in self.modules_to_learn:
                for p in layer.parameters():
                    params.append(p.cpu().numpy().ravel())
        return np.concatenate(params)

    def set_params(self, params):
        assert isinstance(params, np.ndarray)
        ss = 0
        for layer in self.modules_to_learn:
            for p in layer.parameters():
                ee = ss + np.prod(p.shape)
                p.data = torch.from_numpy(
                    params[ss:ee].reshape(p.shape)
                ).float().to(self.device)
                ss = ee
        assert ss == params.size

    def save(self, filename):
        params = self.get_params()
        np.savez(filename, params=params)

    def load(self, filename):
        with np.load(filename) as data:
            params = data['params']
            self.set_params(params)

    def get_num_params(self):
        return self.get_params().size

    def _get_action(self, obs):
        raise NotImplementedError()

    def reset(self):
        pass


@gin.configurable
class MLPSolution(BaseTorchSolution):
    """MLP solution."""

    def __init__(self, device, obs_dim, act_dim, hidden_dim, num_hidden_layers):
        super(MLPSolution, self).__init__(device=device)
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.extend([
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.Tanh(),
            ])
        self.net = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=hidden_dim),
            nn.Tanh(),
            *hidden_layers,
            nn.Linear(in_features=hidden_dim, out_features=act_dim),
            nn.Tanh(),
        ).to(self.device)
        self.modules_to_learn.append(self.net)
        print('device={}, #params={}'.format(
            self.device, self.get_num_params()))

    def _get_action(self, obs):
        x = torch.from_numpy(obs.copy()).float().to(self.device)
        return self.net(x).cpu().numpy()



@gin.configurable
class PIFCSolution(BaseTorchSolution):
    """Permutation invariant solution."""

    def __init__(self,
                 device,
                 act_dim,
                 hidden_dim,
                 msg_dim,
                 pos_em_dim,
                 num_hidden_layers=3,
                 pi_layer_bias=True,
                 pi_layer_scale=True):
        super(PIFCSolution, self).__init__(device=device)
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_em_dim = pos_em_dim
        self.prev_act = torch.zeros(1, self.act_dim)
        #self.prev_act = 0

        self.pi_layer = AttentionNeuronLayer(
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            msg_dim=msg_dim,
            pos_em_dim=pos_em_dim,
            bias=pi_layer_bias,
            scale=pi_layer_scale,
        )
        self.modules_to_learn.append(self.pi_layer)

        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.extend([
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.Tanh(),
            ])
        self.net = nn.Sequential(
            *hidden_layers,
            nn.Linear(in_features=hidden_dim, out_features=act_dim),
            nn.Tanh(), #output layer for scalar values
            #nn.Softmax(), #softmax function added to output "users" probabilities
            
        )
        self.modules_to_learn.append(self.net)

        print('#params={}'.format(self.get_num_params()))

    def _get_action(self, obs):
        x = self.pi_layer(obs=obs, prev_act=self.prev_act)
        #global content code output pass to linear layer (self.net)
        self.prev_act = self.net(x.T)
        
        return self.prev_act.squeeze(0).cpu().numpy()

    def reset(self):
        self.prev_act = torch.zeros(1, self.act_dim)
        self.pi_layer.reset()


