import math
import torch
import time 

from .module import Module, timer

class Add(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)

    def forward(self, x, y):
        return x + y
    

class Mult(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(1)

    def forward(self, x, y):
        return x * y
    

class Bootstrap(Module):
    def __init__(self, input_min, input_max, input_level):
        super().__init__()
        self.input_min = input_min 
        self.input_max = input_max 
        self.input_level = input_level
        self.prescale = 1
        self.postscale = 1
        self.constant = 0

        self.bootstrap_call_times = []
        self.total_bootstrap_time = 0.0
        self.num_bootstrap_calls = 0

    def extra_repr(self):
        l_eff = len(self.scheme.params.get_logq()) - 1
        return f"l_eff={l_eff}"

    def fit(self):
        center = (self.input_min + self.input_max) / 2 
        half_range = (self.input_max - self.input_min) / 2
        low = float(center - self.margin * half_range)
        high = float(center + self.margin * half_range)

        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(
                "Orion generated a non-finite bootstrap range: "
                f"input_range={self.input_range}, "
                f"center={center}, half_range={half_range}, "
                f"low={low}, high={high}"
            )

        self.low = low
        self.high = high

        # We'll want to scale from [A, B] into [-1, 1] using a value of the
        # form 1 / integer, so that way our multiplication back to the range
        # [A, B] (by integer) after bootstrapping doesn't consume a level.
        if self.high - self.low > 2:
            self.postscale = math.ceil((self.high - self.low) / 2)
            self.prescale = 1 / self.postscale

        self.constant = -(self.low + self.high) / 2 

    # CHANGED
    # def compile(self):
    #     # We'll then encode the prescale at the level of the input ciphertext
    #     # to ensure its rescaling is errorless
    #     elements = self.fhe_input_shape.numel()
    #     curr_slots = 2 ** math.ceil(math.log2(elements))

    #     prescale_vec = torch.zeros(curr_slots)
    #     prescale_vec[:elements] = self.prescale

    #     ql = self.scheme.encoder.get_moduli_chain()[self.input_level]
    #     self.prescale_ptxt = self.scheme.encoder.encode(
    #         prescale_vec, level=self.input_level, scale=ql)

    def compile(self):
        elements = self.fhe_input_shape.numel()
        curr_slots = 2 ** math.ceil(math.log2(elements))

        # A plaintext multiplication is only required when:
        #   1. values must actually be prescaled, or
        #   2. unused sparse-bootstrap slots must be zeroed.
        #
        # If elements == curr_slots and prescale == 1, the vector would
        # contain only ones, so multiplying and rescaling is unnecessary.
        self.needs_prescale_multiply = (
            self.prescale != 1 or elements < curr_slots
        )

        self.prescale_ptxt = None

        if self.needs_prescale_multiply:
            prescale_vec = torch.zeros(curr_slots)
            prescale_vec[:elements] = self.prescale

            ql = self.scheme.encoder.get_moduli_chain()[self.input_level]

            self.prescale_ptxt = self.scheme.encoder.encode(
                prescale_vec,
                level=self.input_level,
                scale=ql,
            )

    @timer
    def forward(self, x):
        if not self.he_mode:
            return x

        # Shift into the centered bootstrap interval.
        if self.constant != 0:
            x += self.constant

        # Do not consume a level by multiplying by an all-one plaintext.
        if self.needs_prescale_multiply:
            x *= self.prescale_ptxt

        start = time.perf_counter()
        x = x.bootstrap()
        elapsed = time.perf_counter() - start

        self.bootstrap_call_times.append(elapsed)
        self.total_bootstrap_time += elapsed
        self.num_bootstrap_calls += 1

        if self.postscale != 1:
            x *= self.postscale

        if self.constant != 0:
            x -= self.constant

        return x




