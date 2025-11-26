FROM golang:1.25

WORKDIR /app

COPY . .

# Habilitar ejecución pese a moving GC
ENV ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25

# Proxy de módulos para evitar problemas
ENV GOPROXY=https://proxy.golang.org,direct

RUN go mod tidy

RUN go build -o gorgonia-app .

CMD ["./gorgonia-app"]
