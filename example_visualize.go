// example_visualize.go
package main

import (
	"fmt"
	"log"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/examples/mnist"
	"gorgonia.org/tensor"
)

// Muestra una imagen de MNIST en ASCII y su etiqueta.
func ExampleVisualize() {
	inputs, targets, err := mnist.Load("train", "./testdata", tensor.Float64)
	if err != nil {
		log.Fatal(err)
	}

	// Tomamos la primera imagen y su label
	imgT, err := inputs.Slice(tensor.S(0))
	if err != nil {
		log.Fatal(err)
	}
	lblT, err := targets.Slice(tensor.S(0))
	if err != nil {
		log.Fatal(err)
	}

	img, ok := imgT.(tensor.Tensor)
	if !ok {
		log.Fatal("no pude castear la imagen a tensor.Tensor")
	}
	lbl, ok := lblT.(tensor.Tensor)
	if !ok {
		log.Fatal("no pude castear el label a tensor.Tensor")
	}

	imgData := img.Data().([]float64)
	lblData := lbl.Data().([]float64)

	label := -1
	for i, v := range lblData {
		if v == 1.0 {
			label = i
			break
		}
	}

	fmt.Printf("Etiqueta: %d\n\n", label)

	// La imagen es 28x28 = 784
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			v := imgData[i*28+j]
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

	_ = G.Float64 // para que el import de G no quede “sin usar” si el compilador se pone estricto
}
