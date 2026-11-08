# GPU build stays separate from the previously committed legacy executable.
NVCC ?= nvcc
CXX ?= g++
NVCCFLAGS ?= -O2 -std=c++17
LIBS ?= -lGLEW -lGL -lSDL2 -ltinyxml2 -lpthread

.PHONY: build test clean

build: build/echolocation_game

build/echolocation_game: echolocation_game.cu acoustics.h collision.h svg_raster.h
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ echolocation_game.cu $(LIBS)

test:
	@set -e; for src in tests/test_*.cpp; do \
	    out="/tmp/$$(basename "$$src" .cpp)"; \
	    $(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -pedantic "$$src" -o "$$out"; \
	    "$$out"; \
	done

clean:
	rm -rf build
