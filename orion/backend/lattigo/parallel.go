package main

import (
	"C"
	"sync"
	"unsafe"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
)

// BsgsDiagonalParallel evaluates one baby-step/giant-step diagonal transform
//
//	out = sum_g rot_{g*giantStride}( sum_b babies[b] * pts[g][b] )
//
// with the giant-step groups spread over nWorkers goroutines. Every product is
// MulNew followed by Rescale and the sums run in the same order as the Python
// reference (gqa/ciphertext_orion/ciphertext_ops.evaluate_bsgs_diagonal), so the
// result is bit-for-bit identical; only the wall time changes. Each worker uses
// its own ShallowCopy of the evaluator (Lattigo evaluators carry scratch
// buffers and are not safe to share). Object-heap writes and rotation-key
// generation happen on the calling goroutine only.
//
//export BsgsDiagonalParallel
func BsgsDiagonalParallel(
	babyPtr *C.int, nBaby C.int,
	ptPtr *C.int, nPt C.int,
	sizePtr *C.int, nGroups C.int,
	giantStride C.int, nWorkers C.int,
) C.int {
	babyIDs := unsafe.Slice(babyPtr, int(nBaby))
	ptIDs := unsafe.Slice(ptPtr, int(nPt))
	sizes := unsafe.Slice(sizePtr, int(nGroups))

	babies := make([]*rlwe.Ciphertext, len(babyIDs))
	for i, id := range babyIDs {
		babies[i] = RetrieveCiphertext(int(id))
	}
	pts := make([]*rlwe.Plaintext, len(ptIDs))
	for i, id := range ptIDs {
		pts[i] = RetrievePlaintext(int(id))
	}
	offsets := make([]int, len(sizes))
	total := 0
	for g, s := range sizes {
		offsets[g] = total
		total += int(s)
	}
	if total != len(pts) {
		panic("BsgsDiagonalParallel: group sizes do not match the plaintext count")
	}

	slots := scheme.Params.MaxSlots()
	rotations := []int{}
	for g := 1; g < len(sizes); g++ {
		amount := g * int(giantStride)
		if amount%slots != 0 {
			rotations = append(rotations, amount)
		}
	}
	addRotationKeys(rotations) // single-threaded: may replace scheme.Evaluator's key set

	workers := int(nWorkers)
	if workers < 1 {
		workers = 1
	}
	if workers > len(sizes) {
		workers = len(sizes)
	}
	outputs := make([]*rlwe.Ciphertext, len(sizes))
	jobs := make(chan int, len(sizes))
	for g := range sizes {
		jobs <- g
	}
	close(jobs)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			eval := scheme.Evaluator.ShallowCopy()
			for g := range jobs {
				var acc *rlwe.Ciphertext
				for b := 0; b < int(sizes[g]); b++ {
					prod, err := eval.MulNew(babies[b], pts[offsets[g]+b])
					if err != nil {
						panic(err)
					}
					if err := eval.Rescale(prod, prod); err != nil {
						panic(err)
					}
					if acc == nil {
						acc = prod
					} else if err := eval.Add(acc, prod, acc); err != nil {
						panic(err)
					}
				}
				amount := g * int(giantStride)
				if g > 0 && amount%slots != 0 {
					rot, err := eval.RotateNew(acc, amount)
					if err != nil {
						panic(err)
					}
					acc = rot
				}
				outputs[g] = acc
			}
		}()
	}
	wg.Wait()

	out := outputs[0]
	for g := 1; g < len(outputs); g++ {
		if err := scheme.Evaluator.Add(out, outputs[g], out); err != nil {
			panic(err)
		}
	}
	return C.int(PushCiphertext(out))
}

// BootstrapMany refreshes n independent ciphertexts, each on its own goroutine
// with its own ShallowCopy of the bootstrapper and evaluator (the bootstrapping
// keys are shared read-only). Per ciphertext it does exactly what Bootstrap does,
// so the outputs are identical to n serial calls; results are pushed to the heap
// in input order on the calling goroutine.
//
//export BootstrapMany
func BootstrapMany(idsPtr *C.int, n C.int, numSlots C.int, nWorkers C.int) (*C.int, C.ulong) {
	ids := unsafe.Slice(idsPtr, int(n))
	inputs := make([]*rlwe.Ciphertext, len(ids))
	for i, id := range ids {
		inputs[i] = RetrieveCiphertext(int(id))
	}
	base := GetBootstrapper(int(numSlots))
	postscale := int(1 << (scheme.Params.LogMaxSlots() - base.LogMaxSlots()))

	workers := int(nWorkers)
	if workers < 1 {
		workers = 1
	}
	if workers > len(inputs) {
		workers = len(inputs)
	}
	outputs := make([]*rlwe.Ciphertext, len(inputs))
	jobs := make(chan int, len(inputs))
	for i := range inputs {
		jobs <- i
	}
	close(jobs)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			btp := base.ShallowCopy()
			eval := scheme.Evaluator.ShallowCopy()
			for i := range jobs {
				ctBtp := inputs[i].CopyNew()
				ctBtp.LogDimensions.Cols = btp.LogMaxSlots()
				ctOut, err := btp.Bootstrap(ctBtp)
				if err != nil {
					panic(err)
				}
				if err := eval.Mul(ctOut, postscale, ctOut); err != nil {
					panic(err)
				}
				ctOut.LogDimensions.Cols = scheme.Params.LogMaxSlots()
				outputs[i] = ctOut
			}
		}()
	}
	wg.Wait()

	newIDs := make([]int, len(outputs))
	for i, ct := range outputs {
		newIDs[i] = PushCiphertext(ct)
	}
	return SliceToCArray(newIDs, convertIntToCInt)
}
