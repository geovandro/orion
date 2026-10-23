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
	return bootstrapMany(idsPtr, n, numSlots, nWorkers, -1)
}

// BootstrapManyAtLevel is BootstrapMany with a bootstrapper refreshing to outLevel.
//
//export BootstrapManyAtLevel
func BootstrapManyAtLevel(idsPtr *C.int, n C.int, numSlots C.int, nWorkers C.int, outLevel C.int) (*C.int, C.ulong) {
	return bootstrapMany(idsPtr, n, numSlots, nWorkers, int(outLevel))
}

func bootstrapMany(idsPtr *C.int, n C.int, numSlots C.int, nWorkers C.int, outLevel int) (*C.int, C.ulong) {
	ids := unsafe.Slice(idsPtr, int(n))
	inputs := make([]*rlwe.Ciphertext, len(ids))
	for i, id := range ids {
		inputs[i] = RetrieveCiphertext(int(id))
	}
	base := GetBootstrapperAtLevel(int(numSlots), outLevel)
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

// Batch operation codes for BatchOp.
const (
	batchMulRelinRescale = 0 // a[i] * b[i] (ciphertexts), relinearize, rescale
	batchMulPlainRescale = 1 // a[i] * pt[b[i]], rescale
	batchAdd             = 2 // a[i] + b[i] (ciphertexts)
	batchRotate          = 3 // rotate a[i] left by param
	batchAddPlain        = 4 // a[i] + pt[b[i]]
	batchSubPlain        = 5 // a[i] - pt[b[i]]
)

// BatchOp applies one homomorphic operation elementwise to n independent
// ciphertexts on nWorkers goroutines and returns n new ciphertext ids (inputs
// are not modified). Every element runs exactly what the serial wrappers run
// (MulRelinNew + Rescale, MulNew + Rescale, AddNew, RotateNew, AddNew/SubNew
// with a plaintext), so results are bit-for-bit identical to n serial calls.
// Used for stages made of many same-shaped blocks (the softmax exponential's
// 68 blocks, QK^T). Heap reads/writes and rotation-key generation stay on the
// calling goroutine; each worker owns a ShallowCopy of the evaluator.
//
//export BatchOp
func BatchOp(op C.int, aPtr *C.int, nA C.int, bPtr *C.int, nB C.int, param C.int, nWorkers C.int) (*C.int, C.ulong) {
	aIDs := unsafe.Slice(aPtr, int(nA))
	var bIDs []C.int
	if int(nB) > 0 {
		bIDs = unsafe.Slice(bPtr, int(nB))
		if len(bIDs) != len(aIDs) {
			panic("BatchOp: operand lists differ in length")
		}
	}
	n := len(aIDs)
	as := make([]*rlwe.Ciphertext, n)
	bs := make([]*rlwe.Ciphertext, n)
	ps := make([]*rlwe.Plaintext, n)
	for i := 0; i < n; i++ {
		as[i] = RetrieveCiphertext(int(aIDs[i]))
		switch int(op) {
		case batchMulRelinRescale, batchAdd:
			bs[i] = RetrieveCiphertext(int(bIDs[i]))
		case batchMulPlainRescale, batchAddPlain, batchSubPlain:
			ps[i] = RetrievePlaintext(int(bIDs[i]))
		}
	}
	if int(op) == batchRotate {
		addRotationKeys([]int{int(param)}) // single-threaded
	}

	workers := int(nWorkers)
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}
	outputs := make([]*rlwe.Ciphertext, n)
	jobs := make(chan int, n)
	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			eval := scheme.Evaluator.ShallowCopy()
			for i := range jobs {
				var out *rlwe.Ciphertext
				var err error
				switch int(op) {
				case batchMulRelinRescale:
					out, err = eval.MulRelinNew(as[i], bs[i])
					if err == nil {
						err = eval.Rescale(out, out)
					}
				case batchMulPlainRescale:
					out, err = eval.MulNew(as[i], ps[i])
					if err == nil {
						err = eval.Rescale(out, out)
					}
				case batchAdd:
					out, err = eval.AddNew(as[i], bs[i])
				case batchRotate:
					out, err = eval.RotateNew(as[i], int(param))
				case batchAddPlain:
					out, err = eval.AddNew(as[i], ps[i])
				case batchSubPlain:
					out, err = eval.SubNew(as[i], ps[i])
				default:
					panic("BatchOp: unknown operation")
				}
				if err != nil {
					panic(err)
				}
				outputs[i] = out
			}
		}()
	}
	wg.Wait()

	newIDs := make([]int, n)
	for i, ct := range outputs {
		newIDs[i] = PushCiphertext(ct)
	}
	return SliceToCArray(newIDs, convertIntToCInt)
}
