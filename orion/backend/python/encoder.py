import torch
from .tensors import PlainTensor

class NewEncoder:
    def __init__(self, scheme):
        self.scheme = scheme
        self.params = scheme.params
        self.backend = scheme.backend 
        self.setup_encoder()

    def setup_encoder(self):
        self.backend.NewEncoder()

    def encode(self, values, level=None, scale=None):
        if isinstance(values, list):
            values = torch.tensor(values)
        elif not isinstance(values, torch.Tensor):
            raise TypeError(
                f"Expected 'values' passed to encode() to be a either a list "
                f"or a torch.Tensor, but got {type(values)}.")

        if not level:
            level = self.params.get_max_level()
        if not scale:
            scale = self.params.get_default_scale()

        num_slots = self.params.get_slots()
        num_elements = values.numel()

        values = values.cpu()
        is_complex = torch.is_complex(values)
        pad_length = (-num_elements) % num_slots
        vector = torch.zeros(
            num_elements + pad_length,
            dtype=torch.complex128 if is_complex else torch.get_default_dtype(),
        )
        vector[:num_elements] = values.flatten()
        num_plaintexts = len(vector) // num_slots

        plaintext_ids = []
        for i in range(num_plaintexts):
            chunk = vector[i*num_slots:(i+1)*num_slots]
            if is_complex:
                # Complex packing: both slot halves carry data (EncodeComplex).
                plaintext_id = self.backend.EncodeComplex(
                    chunk.real.tolist(), chunk.imag.tolist(), level, scale)
            else:
                plaintext_id = self.backend.Encode(chunk.tolist(), level, scale)
            plaintext_ids.append(plaintext_id)

        return PlainTensor(self.scheme, plaintext_ids, values.shape)

    def decode(self, plaintensor: PlainTensor):
        values = [] 
        for plaintext_id in plaintensor.ids:
            values.extend(self.backend.Decode(plaintext_id))

        values = torch.tensor(values)[:plaintensor.on_shape.numel()]
        return values.reshape(plaintensor.on_shape)

    def get_moduli_chain(self):
        return self.backend.GetModuliChain()
