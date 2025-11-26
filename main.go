package main

import "fmt"

func main() {
	fmt.Println("Seleccioná un ejemplo para ejecutar:")
	fmt.Println("1. Cargar MNIST")
	fmt.Println("2. Visualizar imagen MNIST")
	fmt.Println("3. Entrenar MLP simple")
	fmt.Println("4. Softmax + CrossEntropy")
	fmt.Println("5. Exportar imagen MNIST a PNG")

	var opt int
	fmt.Scan(&opt)

	switch opt {
	case 1:
		ExampleLoad()
	case 2:
		ExampleVisualize()
	case 3:
		ExampleMLP()
	case 4:
		ExampleLoss()
	case 5:
		ExamplePNG()
	default:
		fmt.Println("Opción inválida")
	}
}
