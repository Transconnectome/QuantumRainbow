# General imports
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import copy

# Qiskit Circuit imports
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import TwoLocal

# Qiskit imports
import qiskit as qk

# Qiskit Machine Learning imports
import qiskit_machine_learning as qkml
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

# PyTorch imports
import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import LBFGS, SGD, Adam, RMSprop
import torch.nn as nn

import torch.nn.functional as F

# from distributed import init_distributed

# Fix seed for reproducibility
seed = 2023
np.random.seed(seed)
torch.manual_seed(seed);

# To get smooth animations on Jupyter Notebooks.
# Note: these plotting function are taken from https://github.com/ageron/handson-ml2
import matplotlib as mpl

import gymnasium as gym
from gymnasium import spaces
import tianshou as ts
import numpy as np

from tianshou.policy import RainbowPolicy
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.utils.net.common import DataParallelNet

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PF")
    parser.add_argument("--ACEgroup", type=str, default="group1")
    parser.add_argument("--log_num", type=str, default='0')        
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-quantiles", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-10.)
    parser.add_argument("--v-max", type=float, default=10.)        
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--target-update-freq", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)    
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--neurons", type=int, default=128)
    parser.add_argument("--qnn-layers", type=int, default=10)
    parser.add_argument("--data-reupload", action="store_true", default=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    return parser.parse_args()

args=get_args()

#functions

def encoding_circuit(inputs, num_qubits = 2, *args):
    """
    Encode classical input data (i.e. the state of the enironment) on a quantum circuit.
    To be used inside the `parametrized_circuit` function.

    Args
    -------
    inputs (list): a list containing the classical inputs.
    num_qubits (int): number of qubits in the quantum circuit.

    Return
    -------
    qc (QuantumCircuit): quantum circuit with encoding gates.

    """

    qc = qk.QuantumCircuit(num_qubits)

    # Encode data with a RX rotation
    for i in range(len(inputs)):
        qc.rx(inputs[i], i)

    return qc


def parametrized_circuit(num_qubits = 2, reuploading = False, reps = 2, insert_barriers = True, meas = False):
    """
    Create the Parameterized Quantum Circuit (PQC) for estimating Q-values.
    It implements the architecure proposed in Skolik et al. arXiv:2104.15084.

    Args
    -------
    num_qubit (int): number of qubits in the quantum circuit.
    reuploading (bool): True if want to use data reuploading technique.
    reps (int): number of repetitions (layers) in the variational circuit.
    insert_barrirerd (bool): True to add barriers in between gates, for better drawing of the circuit.
    meas (bool): True to add final measurements on the qubits.

    Return
    -------
    qc (QuantumCircuit): the full parametrized quantum circuit.
    """

    qr = qk.QuantumRegister(num_qubits, 'qr')
    qc = qk.QuantumCircuit(qr)

    if meas:
        qr = qk.QuantumRegister(num_qubits, 'qr')
        cr = qk.ClassicalRegister(num_qubits, 'cr')
        qc = qk.QuantumCircuit(qr,cr)


    if not reuploading:

        # Define a vector containg Inputs as parameters (*not* to be optimized)
        inputs = qk.circuit.ParameterVector('x', num_qubits)

        # Encode classical input data
        qc.compose(encoding_circuit(inputs, num_qubits = num_qubits), inplace = True)
        if insert_barriers: qc.barrier()

        # Variational circuit
        qc.compose(TwoLocal(num_qubits, ['ry', 'rz'], 'cz', 'circular',
               reps=reps, insert_barriers= insert_barriers,
               skip_final_rotation_layer = True), inplace = True)
        if insert_barriers: qc.barrier()

        # Add final measurements
        if meas: qc.measure(qr,cr)

    elif reuploading:

        # Define a vector containg Inputs as parameters (*not* to be optimized)
        inputs = qk.circuit.ParameterVector('x', num_qubits)

        # Define a vector containng variational parameters
        θ = qk.circuit.ParameterVector('θ', 2 * num_qubits * reps)

        # Iterate for a number of repetitions
        for rep in range(reps):

            # Encode classical input data
            qc.compose(encoding_circuit(inputs, num_qubits = num_qubits), inplace = True)
            if insert_barriers: qc.barrier()

            # Variational circuit (does the same as TwoLocal from Qiskit)
            for qubit in range(num_qubits):
                qc.ry(θ[qubit + 2*num_qubits*(rep)], qubit)
                qc.rz(θ[qubit + 2*num_qubits*(rep) + num_qubits], qubit)
            if insert_barriers: qc.barrier()

            # Add entanglers (this code is for a circular entangler)
            qc.cz(qr[-1], qr[0])
            for qubit in range(num_qubits-1):
                qc.cz(qr[qubit], qr[qubit+1])
            if insert_barriers: qc.barrier()

        # Add final measurements
        if meas: qc.measure(qr,cr)

    return qc

# Select the number of qubits
num_qubits = 2

# Generate the Parametrized Quantum Circuit (note the flags reuploading and reps)
qc = parametrized_circuit(num_qubits = num_qubits,
                          reuploading = args.data_reupload,
                          reps = args.qnn_layers)

# Fetch the parameters from the circuit and divide them in Inputs (X) and Trainable Parameters (params)
# The first four parameters are for the inputs
X = list(qc.parameters)[: num_qubits]

# The remaining ones are the trainable weights of the quantum neural network
params = list(qc.parameters)[num_qubits:]

qc.draw()


# Select a quantum backend to run the simulation of the quantum circuit
# qi = QuantumInstance(qk.Aer.get_backend('statevector_simulator'))

# Create a Quantum Neural Network object starting from the quantum circuit defined above
qnn = SamplerQNN(circuit=qc, input_params=X, weight_params=params)

# Connect to PyTorch
initial_weights = (2*np.random.rand(qnn.num_weights) - 1)
quantum_nn = TorchConnector(qnn, initial_weights)


class encoding_layer(nn.Module):
    def __init__(self, num_qubits=2, init_strategy='uniform'):
        super().__init__()

        self.num_qubits = num_qubits
        self.init_strategy = init_strategy

        if init_strategy == 'uniform':
            self.weights = nn.Parameter(torch.Tensor(num_qubits))
            nn.init.uniform_(self.weights, -1, 1)
        elif init_strategy == 'normal':
            self.weights = nn.Parameter(torch.Tensor(num_qubits))
            nn.init.normal_(self.weights, mean=0, std=1)
        else:
            raise ValueError("Invalid initialization strategy: {}".format(init_strategy))

    def forward(self, x, state=None, info={}):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        x = x.to(self.weights.device)
        x = self.weights * x
        x = torch.atan(x)

        return x

    
class exp_val_layer(nn.Module):
    def __init__(self, action_space=2, init_strategy='uniform'):
        super().__init__()

        self.action_space = action_space
        self.init_strategy = init_strategy

        if init_strategy == 'uniform':
            self.weights = nn.Parameter(torch.Tensor(action_space))
            nn.init.uniform_(self.weights, -1, 1)
        elif init_strategy == 'normal':
            self.weights = nn.Parameter(torch.Tensor(action_space))
            nn.init.normal_(self.weights, mean=0, std=1)
        else:
            raise ValueError("Invalid initialization strategy: {}".format(init_strategy))

        self.mask_ZZ_12 = nn.Parameter(torch.tensor([1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1.]),
                                       requires_grad=False)
        self.mask_ZZ_34 = nn.Parameter(torch.tensor([-1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1.]),
                                       requires_grad=False)

    def forward(self, x, state=None, info={}):
        expval_ZZ_12 = self.mask_ZZ_12 * x
        expval_ZZ_34 = self.mask_ZZ_34 * x

        if len(x.shape) == 1:
            expval_ZZ_12 = torch.sum(expval_ZZ_12)
            expval_ZZ_34 = torch.sum(expval_ZZ_34)
            out = torch.cat((expval_ZZ_12.unsqueeze(0), expval_ZZ_34.unsqueeze(0)))
        else:
            expval_ZZ_12 = torch.sum(expval_ZZ_12, dim=1, keepdim=True)
            expval_ZZ_34 = torch.sum(expval_ZZ_34, dim=1, keepdim=True)
            out = torch.cat((expval_ZZ_12, expval_ZZ_34), 1)

        return self.weights * ((out + 1.) / 2.)
    

class QuantumRainbowNN(torch.nn.Module):
    def __init__(self, state_shape, num_actions, num_quantiles, device, seed=2023):
        super(QuantumRainbowNN, self).__init__()
        self.device = device
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        # Quantum components
        self.encoding = encoding_layer(state_shape)
        self.quantum_nn = TorchConnector(qnn, initial_weights)
        self.exp_val = exp_val_layer(num_actions)

        # Additional fully connected layers for Distributional RL (QRDQN) and Dueling architecture (advantage)
        self.fc_quantiles = nn.Sequential(
            NoisyLinear(num_actions, args.neurons),
            nn.ReLU(),
            NoisyLinear(args.neurons, num_quantiles)
        )
        self.fc_advantage = nn.Sequential(
            NoisyLinear(num_actions, args.neurons),
            nn.ReLU(),
            NoisyLinear(args.neurons, num_actions*num_quantiles)
        )

    def forward(self, x, **kwargs):
        # PQC
        x = self.encoding(x)
        x = self.quantum_nn(x)
        x = self.exp_val(x)

        # Distributional RL: Compute quantiles
        quantiles = self.fc_quantiles(x)
        quantiles = quantiles.view(-1, 1, self.num_quantiles)  # Reshape to [batch_size, num_actions, num_quantiles]

        # Dueling Architecture: Compute advantage
        advantage = self.fc_advantage(x)
        advantage = advantage.view(-1, self.num_actions, self.num_quantiles) # Reshape to [batch_size, num_actions, 1]

        # Compute Final Q-values
        q_values = quantiles + advantage - quantiles.mean(dim=1, keepdim=True)

        return F.softmax(q_values, dim=-1), None

    
    def __deepcopy__(self, memodict={}):
        # Target Network: Create a new instance of the class
        new_instance = QuantumRainbowNN(state_shape=self.state_shape,
                                      num_actions=self.num_actions,
                                      num_quantiles=self.num_quantiles,
                                      device=self.device)

        # Copy the fully connected layers for quantiles and advantage
        new_instance.fc_quantiles = copy.deepcopy(self.fc_quantiles, memodict)
        new_instance.fc_advantage = copy.deepcopy(self.fc_advantage, memodict)

        # Assign the quantum parts after copying
        new_instance.encoding = copy.deepcopy(self.encoding, memodict)
        new_instance.quantum_nn = copy.deepcopy(self.quantum_nn, memodict)
        new_instance.exp_val = copy.deepcopy(self.exp_val, memodict)

        return new_instance


filename= f'/scratch/connectome/justin/{args.ACEgroup}_dataset.csv'

class PFEnv(gym.Env):
    def __init__(self, dataset_path, seed=2023):
        # Load the dataset
        self.df = pd.read_csv(dataset_path)

        # Define the action and observation space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)

        # Initialize the current trial number and participant index
        self.current_trial = 0
        self.current_participant = 0

        # Initialize the environment
        self.reset()
        
    def reset(self):
        # Reset the environment to start a new episode
        self.current_trial = 0
        self.current_participant = 0
        self.total_return = 0

        # Check if there are trials for the current participant
        if self.current_participant not in self.df["Participant.ID"].unique():
            return None

        # Get the participant's data for the current trial
        participant_data = self.df[
            (self.df["Participant.ID"] == self.current_participant) &
            (self.df["Trial_number"] == self.current_trial)
        ]

        # Check if participant_data is empty
        if participant_data.empty:
            return None

        # Get the current state and reward
        state = self._get_state(participant_data)
        reward = participant_data["Apple_return"].values[0]

        return state

    def _get_state(self, data):
        # Check if data is empty
        if data.empty:
            return None

        # Map the environment to an integer (0 or 1)
        state = 0 if data["Environment"].values[0] == "Environment_1" else 1

        return state

    def step(self, action):
        # Execute the chosen action and observe the new state and reward
        self.current_trial += 1

        # Check if the episode is over
        done = False
        if self.current_trial >= len(self.df[self.df["Participant.ID"] == self.current_participant]):
            self.current_trial = 0
            self.current_participant += 1
            done = self.current_participant >= self.df["Participant.ID"].nunique()

        # Get the participant's data for the current trial
        participant_data = self.df[
            (self.df["Participant.ID"] == self.current_participant) &
            (self.df["Trial_number"] == self.current_trial)
        ]

        # Get the next state and reward
        next_state = self._get_state(participant_data)
        reward = participant_data["Apple_return"].values[0]

        # Update the total return
        self.total_return += reward

        return next_state, reward, done, {}

    def render(self, mode='human'):
        # Print the current state and total return
        print(f"Participant: {self.current_participant}, Trial: {self.current_trial}, State: {self.state}, Total Return: {self.total_return}")
        
    def seed(self, seed):
        np.random.seed(seed)
        
env = PFEnv(filename, seed=2023)
train_envs = ts.env.DummyVectorEnv([lambda: PFEnv(filename, seed=2023) for _ in range(args.training_num)])
test_envs = ts.env.DummyVectorEnv([lambda: PFEnv(filename, seed=2023) for _ in range(args.test_num)])

state_shape = env.observation_space.n  # equivalent to 4 for CartPole-v1
action_shape = env.action_space.n  # equivalent to 2 for CartPole-v1


#from tianshou.utils.net.common import Net, DataParallelNet
# import torch.nn.parallel
# from torch.nn import DataParallel
# device = "cuda" if torch.cuda.is_available() else "cpu"

_net = QuantumRainbowNN(state_shape, action_shape, args.num_quantiles, args.device)
net = _net.to(args.device)

#With DP
# net = DataParallel(_net)

#DDP
# args = get_arguments()
# init_distributed(args)
#net = nn.parallel.DistributedDataParallel(_net)

optim = torch.optim.Adam(net.parameters(), lr=args.lr)
policy = RainbowPolicy(net, optim, discount_factor=args.gamma, num_atoms=args.num_quantiles,
                       v_min = args.v_min, v_max = args.v_max,
                       estimation_step=args.n_step,
                       target_update_freq=args.target_update_freq)

policy = policy.to(args.device)



from tianshou.data import PrioritizedVectorReplayBuffer

buffer = PrioritizedVectorReplayBuffer(alpha=args.alpha, beta=args.beta, total_size=args.buffer_size, buffer_num=10)  # max size of the replay buffer

train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)


from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
writer = SummaryWriter(f'log_{args.log_num}/{args.task}_Quantum_Rainbow')
logger = TensorboardLogger(writer)

# Start training
result = offpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    max_epoch=args.epoch,  # maximum number of epochs
    step_per_epoch=args.step_per_epoch,  # number of steps per epoch
    step_per_collect=args.step_per_collect,  # number of steps per data collection
    update_per_step=args.update_per_step,
    episode_per_test=1000,  # number of episodes per test
    batch_size=args.batch_size,  # batch size for updating model
    train_fn=lambda epoch, env_step: policy.set_eps(args.eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(args.eps_test),
    logger=logger)

print(f'Finished training! Use {result["duration"]}')

path = f'/scratch/connectome/justin/{args.task}_Quantum_Rainbow_{args.log_num}.pth'
torch.save(policy.state_dict(), path)

policy.load_state_dict(torch.load(path))
