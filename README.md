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

Reducing the accuracy of the simulation to increase the room size, unfortunately results in poor sound quality very fast. 

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
1. Install:
   sudo apt install build-essential libsdl2-dev libglew-dev nvidia-cuda-toolkit
2. Clone:
   git clone https://github.com/yourusername/echolocation-game.git
   cd echolocation-game
3. Build:
   nvcc -o echolocation_game echolocation_game.cu -lSDL2 -lGL -lGLEW
4. Run:
   ./echolocation_game

### Windows (MSYS2)
1. Install MSYS2 (msys2.org), update:
   pacman -Syu
2. Install dependencies:
   pacman -S mingw-w64-x86_64-toolchain mingw-w64-x86_64-SDL2 mingw-w64-x86_64-glew
   - Install CUDA Toolkit, add nvcc to PATH (e.g., C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin).
3. Clone:
   git clone https://github.com/ballerburg9005/echolocation-game.git
   cd echolocation-game
4. Build:
   nvcc -o echolocation_game.exe echolocation_game.cu -lSDL2 -lGL -lGLEW
5. Run:
   ./echolocation_game.exe

## Controls
- Move: WASD/arrows, strafe: Q/E.
- Cane: Hold left mouse, move to extend.
- Params: 1/2 (pulse), 3/4 (decay), 5/6 (reflect), 7/8 (fudge).
- Toggle objects: Space.

## License
GPLv3. See LICENSE.

## Contributing
Fork, modify, PR. Keep it GPLv3.
