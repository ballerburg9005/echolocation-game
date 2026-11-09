# GPU build stays separate from the previously committed legacy executable.
NVCC ?= nvcc
CXX ?= g++
NVCCFLAGS ?= -O2 -std=c++17
LIBS ?= -lGLEW -lGL -lSDL2 -ltinyxml2 -lpthread

.PHONY: build asmr test clean

build: build/echolocation_game

build/echolocation_game: echolocation_game.cu acoustics.h collision.h svg_raster.h echo_arcade.h visual_mesh.h view3d.h virtual_cane.h contact_audio.h proximity_audio.h source_model.h direct_reference.h soft_ghost.h
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ echolocation_game.cu $(LIBS)

# Separate CUDA/SDL2-only ASMR experiment; no effect on arcade binary.
asmr: build/asmr_room

build/asmr_room: asmr_room.cu asmr_lab_math.h asmr_mp3_audio.h acoustics.h source_model.h
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ asmr_room.cu -lSDL2 -lmpg123 -lpthread

test:
	@set -e; for src in tests/test_*.cpp; do \
	    out="/tmp/$$(basename "$$src" .cpp)"; \
	    $(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -pedantic "$$src" -o "$$out"; \
	    "$$out"; \
	done

clean:
	rm -rf build
