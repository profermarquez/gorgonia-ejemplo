// example_png.go
package main

import (
	"fmt"
	"image"
	"image/png"
	"log"
	"os"

	"gorgonia.org/gorgonia/examples/mnist"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

// Guarda la imagen 0 de MNIST en output.png
func ExamplePNG() {
	inputs, targets, err := mnist.Load("train", "./testdata", tensor.Float64)
	if err != nil {
		log.Fatal(err)
	}

	cols := inputs.Shape()[1] // debería ser 784 = 28x28

	// Arreglo para construir la imagen
	imageBackend := make([]uint8, cols)

	// Construimos la imagen a partir de los valores [0..1] → [0..255]
	for i := 0; i < cols; i++ {
		v, _ := inputs.At(0, i)
		// Escalado manual simple
		val := v.(float64)
		px := uint8(val * 255)
		imageBackend[i] = px
	}

	// Crear imagen 28x28
	img := &image.Gray{
		Pix:    imageBackend,
		Stride: 28,
		Rect:   image.Rect(0, 0, 28, 28),
	}

	// Guardar el archivo PNG
	file, err := os.Create("output.png")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	if err := png.Encode(file, img); err != nil {
		log.Fatal(err)
	}

	// También imprimimos la etiqueta real
	labels, _ := native.MatrixF64(targets.(*tensor.Dense))
	fmt.Println("Label real:", argmax(labels[0]))

	fmt.Println("Imagen guardada como output.png")
}
