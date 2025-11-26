// example_mlp.go
package main

import (
	"fmt"
	"log"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/examples/mnist"
	"gorgonia.org/tensor"
)

func ExampleMLP() {
	const (
		inputSize  = 784
		hiddenSize = 64
		outputSize = 10
		batchSize  = 64
	)

	// 1) Cargar MNIST
	trainX, trainY, err := mnist.Load("train", "./testdata", tensor.Float64)
	if err != nil {
		log.Fatal(err)
	}
	testX, testY, err := mnist.Load("test", "./testdata", tensor.Float64)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Entrenando MLP simple sobre MNIST...")

	g := G.NewGraph()

	// Inputs
	x := G.NewMatrix(g, G.Float64, G.WithShape(batchSize, inputSize), G.WithName("x"))
	y := G.NewMatrix(g, G.Float64, G.WithShape(batchSize, outputSize), G.WithName("y"))

	// Pesos
	w1 := G.NewMatrix(g, G.Float64,
		G.WithShape(inputSize, hiddenSize),
		G.WithName("w1"),
		G.WithInit(G.GlorotN(1.0)),
	)

	b1 := G.NewMatrix(g, G.Float64,
		G.WithShape(1, hiddenSize),
		G.WithName("b1"),
		G.WithInit(G.Zeroes()),
	)

	w2 := G.NewMatrix(g, G.Float64,
		G.WithShape(hiddenSize, outputSize),
		G.WithName("w2"),
		G.WithInit(G.GlorotN(1.0)),
	)

	b2 := G.NewMatrix(g, G.Float64,
		G.WithShape(1, outputSize),
		G.WithName("b2"),
		G.WithInit(G.Zeroes()),
	)

	ones := G.NewMatrix(g, G.Float64,
		G.WithShape(batchSize, 1),
		G.WithInit(G.Ones()),
	)

	// Primera capa
	wx1 := G.Must(G.Mul(x, w1))
	b1exp := G.Must(G.Mul(ones, b1))
	l1pre := G.Must(G.Add(wx1, b1exp))
	l1 := G.Must(G.Rectify(l1pre))

	// Segunda capa
	wx2 := G.Must(G.Mul(l1, w2))
	b2exp := G.Must(G.Mul(ones, b2))
	l2 := G.Must(G.Add(wx2, b2exp))

	probs := G.Must(G.SoftMax(l2, 1))

	// Cross-entropy
	logp := G.Must(G.Log(probs))
	ylogp := G.Must(G.HadamardProd(y, logp))
	summed := G.Must(G.Sum(ylogp, 1))
	neg := G.Must(G.Neg(summed))
	loss := G.Must(G.Mean(neg, 0))

	// Grad
	if _, err := G.Grad(loss, w1, b1, w2, b2); err != nil {
		log.Fatal(err)
	}

	vm := G.NewTapeMachine(g)
	defer vm.Close()

	solver := G.NewAdamSolver(G.WithLearnRate(0.01))
	params := G.Nodes{w1, b1, w2, b2}
	model := G.NodesToValueGrads(params)

	samples := trainX.Shape()[0]

	for i := 0; i+batchSize <= samples; i += batchSize {

		xBatch, _ := trainX.Slice(tensor.S(i, i+batchSize))
		yBatch, _ := trainY.Slice(tensor.S(i, i+batchSize))

		G.Let(x, xBatch.(tensor.Tensor))
		G.Let(y, yBatch.(tensor.Tensor))

		if err := vm.RunAll(); err != nil {
			log.Fatal(err)
		}

		if err := solver.Step(model); err != nil {
			log.Fatal(err)
		}

		vm.Reset()
	}

	fmt.Println("Entrenamiento terminado.")
	fmt.Println("Loss final:", loss.Value())

	// 🔥 AHORA TESTEAMOS UNA IMAGEN
	testOneSample(w1, b1, w2, b2, testX, testY)
}

// --- FUNCIÓN QUE MUESTRA UNA IMAGEN Y SU PREDICCIÓN ---
func testOneSample(w1, b1, w2, b2 *G.Node, testX, testY tensor.Tensor) {
	fmt.Println("\n=== Probando una imagen del test set ===")

	g := G.NewGraph()

	// Un solo sample (batch = 1)
	x := G.NewMatrix(g, G.Float64, G.WithShape(1, 784))
	ones := G.NewMatrix(g, G.Float64, G.WithShape(1, 1), G.WithInit(G.Ones()))

	// W1,b1,W2,b2 como constantes
	w1c := G.NewMatrix(g, G.Float64, G.WithShape(784, 64), G.WithValue(w1.Value()))
	b1c := G.NewMatrix(g, G.Float64, G.WithShape(1, 64), G.WithValue(b1.Value()))
	w2c := G.NewMatrix(g, G.Float64, G.WithShape(64, 10), G.WithValue(w2.Value()))
	b2c := G.NewMatrix(g, G.Float64, G.WithShape(1, 10), G.WithValue(b2.Value()))

	// Forward
	l1 := G.Must(G.Rectify(G.Must(G.Add(G.Must(G.Mul(x, w1c)), G.Must(G.Mul(ones, b1c))))))
	l2 := G.Must(G.Add(G.Must(G.Mul(l1, w2c)), G.Must(G.Mul(ones, b2c))))
	probs := G.Must(G.SoftMax(l2, 1))

	vm := G.NewTapeMachine(g)
	defer vm.Close()

	// Agarramos el sample 0
	x0, _ := testX.Slice(tensor.S(0, 1))
	y0, _ := testY.Slice(tensor.S(0))

	G.Let(x, x0.(tensor.Tensor))

	if err := vm.RunAll(); err != nil {
		log.Fatal(err)
	}

	// Predicción
	p := probs.Value().Data().([]float64)
	pred := argmax(p)

	// Label real
	ydata := y0.(tensor.Tensor).Data().([]float64)
	real := argmax(ydata)

	fmt.Printf("Predicción: %d\n", pred)
	fmt.Printf("Real:       %d\n\n", real)

	fmt.Println("Imagen:")
	printASCII(x0.(tensor.Tensor).Data().([]float64))
}

func argmax(v []float64) int {
	max := v[0]
	idx := 0
	for i := 1; i < len(v); i++ {
		if v[i] > max {
			max = v[i]
			idx = i
		}
	}
	return idx
}

func printASCII(img []float64) {
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			v := img[i*28+j]
			ch := ' '
			switch {
			case v > 0.8:
				ch = '#'
			case v > 0.5:
				ch = '*'
			case v > 0.2:
				ch = '.'
			default:
				ch = ' '
			}
			fmt.Printf("%c", ch)
		}
		fmt.Println()
	}
}
