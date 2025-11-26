package main

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia/examples/mnist"
	"gorgonia.org/tensor"
)

func ExampleLoad() {
	dataDir := "./testdata"

	for _, typ := range []string{"test", "train"} {
		inputs, targets, err := mnist.Load(typ, dataDir, tensor.Float64)
		if err != nil {
			log.Fatalf("Error cargando %s: %v", typ, err)
		}
		fmt.Println()
		fmt.Println("=== ", typ, " ===")
		fmt.Println("inputs:", inputs.Shape())
		fmt.Println("labels:", targets.Shape())
	}
}
