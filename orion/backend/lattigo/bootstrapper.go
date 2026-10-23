package main

import (
	"C"
)
import (
	"fmt"
	"math"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
	"github.com/baahl-nyu/lattigo/v6/utils"
)

// A bootstrapper is identified by its slot count and its output level.
// outLevel < 0 means the scheme's maximum level (the original behaviour).
type btpKey struct {
	slots    int
	outLevel int
}

// Map to store bootstrapping.Evaluators by slot count and output level.
var bootstrapperMap = make(map[btpKey]*bootstrapping.Evaluator)

func normalizeLevel(outLevel int) int {
	if outLevel < 0 || outLevel >= scheme.Params.MaxLevel() {
		return -1
	}
	return outLevel
}

// residualParameters returns the scheme parameters cut to the first
// outLevel + 1 primes. A bootstrapper built on them refreshes to outLevel,
// and since the primes are a prefix of the scheme's chain its output is an
// ordinary level-outLevel ciphertext of the scheme. Every step of that
// bootstrap carries fewer limbs, so it is cheaper.
func residualParameters(outLevel int) ckks.Parameters {
	if outLevel < 0 {
		return *scheme.Params
	}
	lit := scheme.Params.ParametersLiteral()
	lit.Q = scheme.Params.Q()[:outLevel+1]
	lit.P = scheme.Params.P()
	lit.LogQ = nil
	lit.LogP = nil
	params, err := ckks.NewParametersFromLiteral(lit)
	if err != nil {
		panic(err)
	}
	return params
}

func newBootstrapper(logP []int, slots, outLevel int) {
	key := btpKey{slots, normalizeLevel(outLevel)}
	if _, exists := bootstrapperMap[key]; exists {
		return
	}
	residual := residualParameters(key.outLevel)

	btpParametersLit := bootstrapping.ParametersLiteral{
		LogN:     utils.Pointy(residual.LogN()),
		LogP:     logP,
		Xs:       residual.Xs(),
		LogSlots: utils.Pointy(int(math.Log2(float64(slots)))),
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(residual, btpParametersLit)
	if err != nil {
		panic(err)
	}

	// The keys extend the secret's first limb, so the scheme's secret key
	// serves residual chains of any length.
	btpKeys, _, err := btpParams.GenEvaluationKeys(scheme.SecretKey)
	if err != nil {
		panic(err)
	}

	var btpEval *bootstrapping.Evaluator
	if btpEval, err = bootstrapping.NewEvaluator(btpParams, btpKeys); err != nil {
		panic(err)
	}
	bootstrapperMap[key] = btpEval
}

//export NewBootstrapper
func NewBootstrapper(
	LogPs *C.int,
	lenLogPs C.int,
	numSlots C.int,
) {
	logP := CArrayToSlice(LogPs, lenLogPs, convertCIntToInt)
	newBootstrapper(logP, int(numSlots), -1)
}

// NewBootstrapperAtLevel builds a bootstrapper whose output level is
// outLevel instead of the scheme's maximum level.
//
//export NewBootstrapperAtLevel
func NewBootstrapperAtLevel(
	LogPs *C.int,
	lenLogPs C.int,
	numSlots C.int,
	outLevel C.int,
) {
	logP := CArrayToSlice(LogPs, lenLogPs, convertCIntToInt)
	newBootstrapper(logP, int(numSlots), int(outLevel))
}

func bootstrapOne(ciphertextID, numSlots, outLevel int) int {
	ctIn := RetrieveCiphertext(ciphertextID)
	bootstrapper := GetBootstrapperAtLevel(numSlots, outLevel)

	ctBtp := ctIn.CopyNew()
	ctBtp.LogDimensions.Cols = bootstrapper.LogMaxSlots()

	ctOut, err := bootstrapper.Bootstrap(ctBtp)
	if err != nil {
		panic(err)
	}

	postscale := int(1 << (scheme.Params.LogMaxSlots() - bootstrapper.LogMaxSlots()))
	scheme.Evaluator.Mul(ctOut, postscale, ctOut)

	ctOut.LogDimensions.Cols = scheme.Params.LogMaxSlots()
	return PushCiphertext(ctOut)
}

//export Bootstrap
func Bootstrap(ciphertextID, numSlots C.int) C.int {
	return C.int(bootstrapOne(int(ciphertextID), int(numSlots), -1))
}

//export BootstrapAtLevel
func BootstrapAtLevel(ciphertextID, numSlots, outLevel C.int) C.int {
	return C.int(bootstrapOne(int(ciphertextID), int(numSlots), int(outLevel)))
}

func GetBootstrapper(numSlots int) *bootstrapping.Evaluator {
	return GetBootstrapperAtLevel(numSlots, -1)
}

func GetBootstrapperAtLevel(numSlots, outLevel int) *bootstrapping.Evaluator {
	key := btpKey{numSlots, normalizeLevel(outLevel)}
	bootstrapper, exists := bootstrapperMap[key]
	if !exists {
		panic(fmt.Errorf("no bootstrapper found for slot count %d, output level %d", numSlots, outLevel))
	}
	return bootstrapper
}

//export DeleteBootstrappers
func DeleteBootstrappers() {
	bootstrapperMap = make(map[btpKey]*bootstrapping.Evaluator)
}
