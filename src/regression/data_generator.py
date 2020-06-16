"""
Code for generating and loading data
"""

import numpy as np
import os
import random
import tensorflow as tf




## Copied from Finn's implementation https://github.com/cbfinn/maml/blob/master/data_generator.py
class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self,  num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """

        self.num_samples_per_class = num_samples_per_class
        self.batch_size = batch_size



        self.generate = self.generate_sinusoid_batch
        self.amp_range = config.get('amp_range', [0.1, 5.0])
        self.omega_range = config.get('ome_range', [0.8, 1.2])
        self.phase_range = config.get('phase_range', [0, np.pi])
        self.input_range = config.get('input_range', [-5.0, 5.0])
        self.dim_input = 1
        self.dim_output = 1

    def generate_sinusoid_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        omega = np.random.uniform(self.omega_range[0], self.omega_range[1], [self.batch_size])
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(omega[func] * init_inputs[func]-phase[func])
        return init_inputs, outputs, amp, phase, omega
