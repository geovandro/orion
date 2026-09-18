import sys
import math

class PlainTensor:
    def __init__(self, scheme, ptxt_ids, shape, on_shape=None):
        self.scheme = scheme
        self.backend = scheme.backend
        self.encoder = scheme.encoder
        
        self.ids = [ptxt_ids] if isinstance(ptxt_ids, int) else ptxt_ids
        self.shape = shape 
        self.on_shape = on_shape or shape

    def __del__(self):
        if 'sys' in globals() and sys.modules and self.scheme:
            try:
                for idx in self.ids:
                    self.backend.DeletePlaintext(idx)
            except Exception: 
                pass # avoids errors for GC at program termination

    def __len__(self):
        return len(self.ids)
    
    def __str__(self):
        return str(self.decode())
    
    def mul(self, other, in_place=False):
        if not isinstance(other, CipherTensor):
            raise ValueError(f"Multiplication between PlainTensor and "
                             f"{type(other)} is not supported.")

        mul_ids = []
        for i in range(len(self.ids)):
            mul_id = self.evaluator.mul_ciphertext(
                other.ids[i], self.ids[i], in_place)
            mul_ids.append(mul_id)

        if in_place:
            return other
        return CipherTensor(self.scheme, mul_ids, self.shape, self.on_shape) 

    def __mul__(self, other):
        return self.mul(other, in_place=False)     

    def __imul__(self, other):
        return self.mul(other, in_place=True)
    
    def _check_valid(self, other):
        return 
        
    def get_ids(self):
        return self.ids
    
    def scale(self):
        return self.backend.GetPlaintextScale(self.ids[0])
    
    def set_scale(self, scale):
        for ptxt in self.ids:
            self.backend.SetPlaintextScale(ptxt, scale)

    def level(self):
        return self.backend.GetPlaintextLevel(self.ids[0])
    
    def slots(self):
        return self.backend.GetPlaintextSlots(self.ids[0])
    
    def min(self):
        return self.decode().min()
    
    def max(self):
        return self.decode().max()
    
    def moduli(self):
        return self.backend.GetModuliChain()
    
    def decode(self):
        return self.encoder.decode(self)
    

class CipherTensor:
    def __init__(self, scheme, ctxt_ids, shape, on_shape=None):
        self.scheme = scheme
        self.backend = scheme.backend 
        self.encryptor = scheme.encryptor
        self.evaluator = scheme.evaluator
        self.bootstrapper = scheme.bootstrapper

        self.ids = [ctxt_ids] if isinstance(ctxt_ids, int) else ctxt_ids 
        self.shape = shape 
        self.on_shape = on_shape or shape

    def __del__(self):
        if 'sys' in globals() and sys.modules and self.scheme:
            try:
                for idx in self.ids:
                    self.backend.DeleteCiphertext(idx)
            except Exception: 
                pass # avoids errors for GC at program termination

    def __len__(self):
        return len(self.ids)
    
    def __str__(self):
        ptxt = self.decrypt()
        return str(ptxt.decode())
    
    #--------------#
    #  Operations  #
    #--------------#
    
    def __neg__(self):
        neg_ids = []
        for ctxt in self.ids:
            neg_id = self.evaluator.negate(ctxt)
            neg_ids.append(neg_id)

        return CipherTensor(self.scheme, neg_ids, self.shape, self.on_shape)
    
    def add(self, other, in_place=False):
        self._check_valid(other)

        add_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                add_id = self.evaluator.add_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                add_id = self.evaluator.add_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                add_id = self.evaluator.add_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Addition between CipherTensor and "
                                 f"{type(other)} is not supported.")

            add_ids.append(add_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, add_ids, self.shape, self.on_shape)
    
    def __add__(self, other):
        return self.add(other, in_place=False)
    
    def __radd__(self, other):
        return self.add(other, in_place=False)

    def __iadd__(self, other):
        return self.add(other, in_place=True)
    
    def sub(self, other, in_place=False):
        self._check_valid(other)

        sub_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                sub_id = self.evaluator.sub_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                sub_id = self.evaluator.sub_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                sub_id = self.evaluator.sub_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Subtraction between CipherTensor and "
                                 f"{type(other)} is not supported.")

            sub_ids.append(sub_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, sub_ids, self.shape, self.on_shape)
    
    def __sub__(self, other):
        return self.sub(other, in_place=False)

    def __isub__(self, other):
        return self.sub(other, in_place=True)

    def __rsub__(self, other):
        return self.sub(other, in_place=False)
    
    def mul(self, other, in_place=False):
        self._check_valid(other)

        mul_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                mul_id = self.evaluator.mul_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                mul_id = self.evaluator.mul_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                mul_id = self.evaluator.mul_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Multiplication between CipherTensor and "
                                 f"{type(other)} is not supported.")
            
            mul_ids.append(mul_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, mul_ids, self.shape, self.on_shape) 
    
    def __mul__(self, other):
        return self.mul(other, in_place=False)     

    def __imul__(self, other):
        return self.mul(other, in_place=True)

    def __rmul__(self, other):
        return self.mul(other, in_place=False)
    
    def roll(self, amount, in_place=False):
        rot_ids = []
        for ctxt in self.ids:
            rot_id = self.evaluator.rotate(ctxt, amount, in_place)
            rot_ids.append(rot_id)

        return CipherTensor(self.scheme, rot_ids, self.shape, self.on_shape)

    def roll_many(self, amounts):
        """Rotate each ciphertext by several amounts using hoisting."""
        amounts = [int(amount) for amount in amounts]
        if not amounts:
            return []

        ids_by_rotation = [[] for _ in amounts]
        for ctxt_id in self.ids:
            rotated_ids = self.evaluator.rotate_many(ctxt_id, amounts)
            for index, rotated_id in enumerate(rotated_ids):
                ids_by_rotation[index].append(rotated_id)

        return [
            CipherTensor(self.scheme, ids, self.shape, self.on_shape)
            for ids in ids_by_rotation
        ]

    def mul_raw(self, other):
        """Multiply ciphertexts without relinearization or rescaling."""
        self._check_valid(other)
        if not isinstance(other, CipherTensor):
            raise ValueError("mul_raw requires another CipherTensor.")
        if len(self.ids) != len(other.ids):
            raise ValueError(
                "CipherTensor objects must contain the same number of ciphertexts."
            )

        output_ids = [
            self.evaluator.mul_ciphertext_raw(left_id, right_id)
            for left_id, right_id in zip(self.ids, other.ids)
        ]
        return CipherTensor(self.scheme, output_ids, self.shape, self.on_shape)

    def relinearize(self):
        output_ids = [
            self.evaluator.relinearize(ctxt_id) for ctxt_id in self.ids
        ]
        return CipherTensor(self.scheme, output_ids, self.shape, self.on_shape)

    # NEW 
    def rescale(self):
        output_ids = [
            self.evaluator.rescale(ctxt_id, in_place=False)
            for ctxt_id in self.ids
        ]
        return CipherTensor(self.scheme, output_ids, self.shape, self.on_shape)

    # TEMP - NEW: Diagnostic wrapper for Orion-assigned module input levels.
    def drop_level(self, levels):
        levels = int(levels)
        if levels < 0:
            raise ValueError("levels must be non-negative")
        if levels == 0:
            return self
        if levels > self.level():
            raise ValueError(
                f"Cannot drop {levels} levels from ciphertext at level "
                f"{self.level()}."
            )
        output_ids = [
            self.evaluator.drop_level(ctxt_id, levels)
            for ctxt_id in self.ids
        ]
        return CipherTensor(self.scheme, output_ids, self.shape, self.on_shape)

    def conjugate(self, in_place=False):
        """Complex-conjugate each slot of this ciphertext.

        For a purely real ciphertext this is a no-op (up to CKKS noise).
        For a ciphertext packing two independent real lanes as
        re + im*1j, conjugate() negates the imaginary lane, which is
        useful for de-interleaving packed lanes (e.g. combined with a
        multiplication to extract re(a)*re(b) + im(a)*im(b) without
        cross terms).
        """
        conj_ids = []
        for ctxt in self.ids:
            conj_id = self.evaluator.conjugate(ctxt, in_place)
            conj_ids.append(conj_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, conj_ids, self.shape, self.on_shape)

    def mul_by_i(self, in_place=False):
        """Multiply every slot by the imaginary unit i.

        i is a Gaussian integer, so Lattigo's CKKS evaluator treats this
        as scale-preserving -- no rescale is performed, and no CKKS level
        is consumed (mirrors the existing integer scalar-multiply
        optimization).
        """
        mul_ids = []
        for ctxt in self.ids:
            mul_id = self.evaluator.mul_by_i(ctxt, in_place)
            mul_ids.append(mul_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, mul_ids, self.shape, self.on_shape)

    def pack_complex(self, other):
        """Pack two same-shape real-valued CipherTensors into one complex
        ciphertext z = self + i*other.

        Both the imaginary-lane injection (mul_by_i) and the ciphertext
        addition are level-free, so z ends up at the same CKKS level as
        self/other. This lets two ciphertexts share a single bootstrap
        call instead of requiring one each.
        """
        if not isinstance(other, CipherTensor):
            raise ValueError("pack_complex requires another CipherTensor.")
        if len(self.ids) != len(other.ids):
            raise ValueError(
                "CipherTensor objects must contain the same number of ciphertexts."
            )

        i_other = other.mul_by_i(in_place=False)
        return self.add(i_other, in_place=False)

    def unpack_complex(self):
        """Split a complex-packed ciphertext z = a + i*b back into its
        real and imaginary lanes: (a, b).

        a = Re(z) = (z + conj(z)) / 2
        b = Im(z) = -i * (z - conj(z)) / 2

        conj(z) and mul_by_i are level-free; only the final scalar
        multiply by 0.5 costs one rescale (1 CKKS level), which happens
        right after a bootstrap where level budget is abundant.
        """
        conj = self.conjugate(in_place=False)

        real_part = self.add(conj, in_place=False).mul(0.5, in_place=False)

        imag_diff = self.sub(conj, in_place=False)
        imag_part = imag_diff.mul_by_i(in_place=False).mul(-0.5, in_place=False)

        return real_part, imag_part

    def _check_valid(self, other):
        return
    
    #----------------------
    #
    #---------------------
    
    def scale(self):
        return self.backend.GetCiphertextScale(self.ids[0])
    
    def set_scale(self, scale):
        for ctxt in self.ids:
            self.backend.SetCiphertextScale(ctxt, scale)

    def level(self):
        return self.backend.GetCiphertextLevel(self.ids[0])
    
    def slots(self):
        return self.backend.GetCiphertextSlots(self.ids[0])
    
    def degree(self):
        return self.backend.GetCiphertextDegree(self.ids[0])
    
    def min(self):
        return self.decrypt().min()
    
    def max(self):
        return self.decrypt().max()
    
    def moduli(self):
        return self.backend.GetModuliChain()
    
    def bootstrap(self):
        elements = self.on_shape.numel()
        slots = 2 ** math.ceil(math.log2(elements))
        slots = int(min(self.slots(), slots)) # sparse bootstrapping
        
        btp_ids = []
        for ctxt in self.ids:
            btp_id = self.bootstrapper.bootstrap(ctxt, slots)
            btp_ids.append(btp_id)

        return CipherTensor(self.scheme, btp_ids, self.shape, self.on_shape)
        
    def decrypt(self):
        return self.encryptor.decrypt(self)
