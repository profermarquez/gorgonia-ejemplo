## Requerimientos para el ejemplo https://gorgonia.org/tutorials/mnist/
go install gorgonia.org/gorgonia@latest


## Para compilar el ejemplo en docker
docker build -t gorgonia-test .

docker run -it --entrypoint sh gorgonia-test

## Para descargar el dataset y renombrar los archivos

https://github.com/fgnt/mnist

Rename-Item "t10k-labels.idx1-ubyte" "t10k-labels-idx1-ubyte"

Rename-Item "t10k-images.idx3-ubyte" "t10k-images-idx3-ubyte"

Rename-Item "train-labels.idx1-ubyte" "train-labels-idx1-ubyte"

Rename-Item "train-images.idx3-ubyte" "train-images-idx3-ubyte"


## Demas ejemplos   
https://gorgonia.org/tutorials/iris/
