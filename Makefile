# CUDA/SDL2 arcade and ASMR experiment build independently.
NVCC ?= nvcc
CXX ?= g++
NVCCFLAGS ?= -O2 -std=c++17
LIBS ?= -lGLEW -lGL -lSDL2 -ltinyxml2 -lpthread

.PHONY: build asmr test clean

build: build/echolocation_game

build/echolocation_game: echolocation_game.cu acoustics.h collision.h svg_raster.h echo_arcade.h visual_mesh.h view3d.h virtual_cane.h contact_audio.h proximity_audio.h source_model.h direct_reference.h soft_ghost.h
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ echolocation_game.cu $(LIBS)

asmr: build/asmr_room

# librsvg is HOST-C++ code, not CUDA: nvcc's GLib macro parser breaks
# if these headers are included from within a .cu file.
build/asmr_svg_geometry.o: asmr_svg_geometry.cpp asmr_svg_geometry.h asmr_lab_math.h
	mkdir -p build
	$(CXX) -std=c++17 -O2 $(shell pkg-config --cflags librsvg-2.0 cairo) -c -o $@ asmr_svg_geometry.cpp

build/asmr_room: asmr_room.cu asmr_lab_math.h asmr_dynamic_head.h asmr_mp3_audio.h asmr_svg_geometry.h build/asmr_svg_geometry.o acoustics.h source_model.h
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ asmr_room.cu build/asmr_svg_geometry.o -lSDL2 -lmpg123 $(shell pkg-config --libs librsvg-2.0 cairo) -lpthread

test:
	@set -e; for src in tests/test_*.cpp; do \
	    out="/tmp/$$(basename "$$src" .cpp)"; \
	    if [ "$(basename "$src")" = "test_asmr_mp3.cpp" ]; then \
	        $(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
	            "$src" -o "$out" -lSDL2 -lmpg123; \
	    elif [ "$(basename "$src")" = "test_asmr_svg.cpp" ]; then \
	        $(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
	            $$(pkg-config --cflags librsvg-2.0 cairo) "$$src" \
	            asmr_svg_geometry.cpp -o "$$out" \
	            $$(pkg-config --libs librsvg-2.0 cairo); \
	    else \
	        $(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
	            "$$src" -o "$$out"; \
	    fi; \
	    "$$out"; \
	done

clean:
	rm -rf build
