// example_loss.go
package main

import (
	"fmt"
	"log"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Ejemplo sencillo de Softmax + cross-entropy con la API nueva.
func ExampleLoss() {
	g := G.NewGraph()

	// logits y targets para 2 muestras y 3 clases
	logits := G.NewMatrix(g, G.Float64, G.WithShape(2, 3), G.WithName("logits"))
	targets := G.NewMatrix(g, G.Float64, G.WithShape(2, 3), G.WithName("targets"))

	// softmax sobre axis=1 (clases)
	probs := G.Must(G.SoftMax(logits, 1))

	// cross entropy: mean(-sum(y * log(p), axis=1))
	logp := G.Must(G.Log(probs))
	yLogP := G.Must(G.HadamardProd(targets, logp))
	summed := G.Must(G.Sum(yLogP, 1)) // sum sobre clases
	neg := G.Must(G.Neg(summed))
	loss := G.Must(G.Mean(neg, 0)) // promedio sobre batch

	// Gradientes respecto de logits (para mostrar que corre)
	if _, err := G.Grad(loss, logits); err != nil {
		log.Fatal(err)
	}

	vm := G.NewTapeMachine(g)
	defer vm.Close()

	logitsVal := tensor.New(
		tensor.WithShape(2, 3),
		tensor.WithBacking([]float64{
			1, 2, 3,
			1, 2, 3,
		}),
	)

	targetsVal := tensor.New(
		tensor.WithShape(2, 3),
		tensor.WithBacking([]float64{
			0, 0, 1, // sample 1 -> clase 2
			0, 1, 0, // sample 2 -> clase 1
		}),
	)

	G.Let(logits, logitsVal)
	G.Let(targets, targetsVal)

	if err := vm.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Loss: %v\n", loss.Value())
}
