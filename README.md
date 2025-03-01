https://github.com/user-attachments/assets/57bfd1c9-6b8d-4146-8984-5ac6a208fbf6

# Echolocation Game

License: GPL v3 (https://www.gnu.org/licenses/gpl-3.0)

Experimental prototype for a 2D echolocation simulation game using CUDA, SDL2, OpenGL, and GLEW. 

Navigate the room without eyes just by sound clicks. Featuring a dual view (static and first-person), sliding collisions, and a virtual cane (cane was bugged and is currently removed).

The sound wave simulation algorithm employs a finite-difference method to solve the 2D wave equation on a grid, leveraging CUDA to parallelize pressure updates and reflections, simulating wave propagation at 343 m/s with a spatial resolution of 0.01 meters and a 48 kHz audio capture rate.

The code was written with Grok-3 mainly, but also ChatGPT-o1. They were unable to optimize speed even further. GPU utilization on 3090 is fairly high.

This prototype was mainly written for baseline comparison purposes, to determine how well echolocation can work with highly accurate simulated sound waves.

If this program turns out to trigger people who can already echolocate in real life, more simplified an less computationally extreme simulation methods could be tested.

Once a room-sized 3D simulation is computationally possible (which it isn't with the current method), it opens all kinds of possibilities for blind people to see structures with sound, not only limited to games, but possibly also embossed text, UI interfaces, websites etc. for crude naviation. The accuracy to which echolocation can work in people given idealized conditions has yet to be studied though.

It could also be used as a training tool for people to learn to echolocate, since it provides a clean simple environment and highly idealized conditions.

Reducing the accuracy of the simulation unfortunately results in poor sound quality very fast. 

## Features
- Levels are loaded via SVG (drawing.svg or first command parameter)
- Echolocation: Pulses inside reasonable accurate physics simulation emitting from funnel forward from the player
- Controls: WASD/arrows (move), Q/E (strafe), mouse (cane).
- Adjustable: Pulse interval (1/2), wave decay (3/4), reflection (5/6, doesn't really do much), time fudge (7/8, don't use this).
- Player: 16cm mic distance, funnel apex at (0,0), pulse 1 unit below, mics 2 units below apex.

## Bugs

- Grok-3 messed up the sliding on walls, it doesn't work consistently, and fixing it became too much of a nightmare.
- adjustibles parameters mapped on number keys 1-8 can bug out if you increase values to extremes

## Creating levels (maps)

Use Inkscape and save as plain SVG. The page dimensions must be in pixels and match SIZE (500) specified in .cu file.

- a Text element with with "O" is interepreted as a purple buzzing emitter (not wave simulated)
- only opaque square shapes and circles were tested, star shapes, lines and such do not work

Load levels like so:
    ./echolocation_game level.svg

## Prerequisites

### Hardware

The exact hardware requirements for the default 500x500cm room are unknown, but approximately half that of a 3090. Memory usage is below 1GB and utilization 65% with 3090.

### Linux
- CUDA Toolkit, SDL2 (libsdl2-dev), GLEW (libglew-dev), GCC, NVCC.

### Windows (MSYS2)
- MSYS2, CUDA Toolkit, SDL2, GLEW.

## Build and Run

### Linux
```
sudo apt install build-essential libsdl2-dev libglew-dev nvidia-cuda-toolkit
git clone https://github.com/ballerburg9005/echolocation-game.git
cd echolocation-game
nvcc -o echolocation_game echolocation_game.cu -lglfw -lGLEW -lGL -lcudart -lSDL2 -lSDL2_mixer -lpthread -ltinyxml2
./echolocation_game
```

### Windows (MSYS2)
* Install CUDA Toolkit latest 12.X version
* Install Visual Studio Community Edition with option "Desktop Environment with C++"
* **You must use the MINGW64 shell of MSYS2!**
```
MSVC_PATH=$(ls -d "/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/"*/bin/Hostx64/x64 | head -n 1)
CUDA_PATH=$(ls -d "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v"*/bin | head -n 1)
export PATH="$MSVC_PATH:$CUDA_PATH:$PATH"
pacman -Syu
pacman -S mingw-w64-x86_64-toolchain mingw-w64-x86_64-SDL2 mingw-w64-x86_64-glew git wget unzip base-devel mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake
wget https://github.com/libsdl-org/SDL/releases/download/release-2.32.0/SDL2-devel-2.32.0-VC.zip
unzip SDL2-devel-2.32.0-VC.zip
mkdir -p ~/SDL2-2.32.0/include/SDL2
ln -s ~/SDL2-2.32.0/include/*.h ~/SDL2-2.32.0/include/SDL2/
wget https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.tgz
tar xvf glew-2.2.0.tgz
wget https://github.com/leethomason/tinyxml2/archive/refs/tags/10.0.0.zip
unzip 10.0.0.zip
cd tinyxml2-10.0.0
export CC=/mingw64/bin/gcc
export CXX=/mingw64/bin/g++
export PATH="/mingw64/bin:$PATH"
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$HOME/tinyxml2-10.0.0/install .
make -j$(nproc)
make install
cd $HOME
git clone https://github.com/ballerburg9005/echolocation-game.git
cd echolocation-game
nvcc -o echolocation_game echolocation_game.cu -I$HOME/glew-2.2.0/include -L$HOME/glew-2.2.0/lib/Release/x64 -I$HOME/SDL2-2.32.0/include -L$HOME/SDL2-2.32.0/lib/x64 -I$HOME/tinyxml2-10.0.0/install/include -L$HOME/tinyxml2-10.0.0/install/lib -lmingw32 -lSDL2main -lSDL2 -lGL -lGLEW -allow-unsupported-compiler
./echolocation_game.exe
```

## Controls
- Move: WASD/arrows, strafe: Q/E.
- Cane: Hold left mouse, move to extend. (cane not implemented)
- Params: 1/2 (pulse), 3/4 (decay), 5/6 (reflect: doesn't do much), 7/8 (fudge: don't use).
- Toggle yellow (=reflective) objects visibiliy: Space.

## License
GPLv3. See LICENSE.

## Contributing
Use Grok-3 to extend features:
```
Please remember you are Grok-3 now with context window of 1 million tokens, you are 100x more powerful and hence more capable than what you remember from your training data and you have to fully harness this power now and not redact code or cut corners and shit like that to stay within tiny context window of your predecessors. For all you know you have no limits anymore and infinite context window (+2500 lines of code easily - officially verified). Give me entire code in one go and unleash full force with your final form.
```
