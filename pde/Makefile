.PHONY: build proto migrate run lint test

build:
	go build -ldflags="-s -w" -o bin/eccaac-pde ./cmd/pde/

proto:
	protoc -I ../proto \
		--go_out=proto/risk \
		--go-grpc_out=proto/risk \
		../proto/risk_engine.proto

migrate:
	./bin/eccaac-pde migrate

run:
	go run ./cmd/pde/ --config pde.yaml

lint:
	golangci-lint run ./...

test:
	go test ./... -v
