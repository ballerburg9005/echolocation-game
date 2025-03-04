https://github.com/user-attachments/assets/57bfd1c9-6b8d-4146-8984-5ac6a208fbf6

*The video shows the game being used. There are two 500px square viewports rendered side-by-side. The impression of the graphics is very simple like in early computer games such as Pac-Man. The full scene on the left-side viewport can easily be described: there is a vertical very thick wall about 1/4th to the left with an opening about 1/3rd from the bottom. In the top left corner is a purple dot. The right hand room has 3 medium-sized circles lined up vertically around the center on the right side spaced rougly with similar distance but higher distance to the boundaries of the level. There is a slightly larger almost square rectangle in the upper half of the entire level almost center but slightly more downwards. The player is near the bottom left corner facing downwards. The player is composed of two green dots representing the virtual microphones, with a yellow hollow wedge in the middle that has a open end in the downward direction (i.e. two lines with one common origin point diverging roughly by 45 degrees). Inside that wedge is a red dot which is the sound pulse origin. The right-side viewport renders the same scene relative to the player's rotation fixed in the middle facing always upwards. The viewport is zoomed in considerably (4x) and there is as much spacing between the pixels, indicating that one pixel corresponds to the resolution of the FDTD wave simulation (1px=1cm). During the simulation sound waves flash up on the screen from the pulses now and then, but due to the speed of sound being much higher than frame rate, they are only weakly and momentarily visible. The borders of the level as well as the objects inside it and the wedge are yellow, indicating that they reflect sound waves. The background is black. From 0s to 14s the player orients himself and moves towards the front of the rectangle near the middle, continuing until 19s to move back and forth towards it, he then moves clockwise around it to face the upper circle from the left side and bumps into it. From 28s on he slaloms around the next circle while bumping into the upper and lower circle and and the nearby wall. From 35s he walks through the door and bumps into the left side of it at 41s, goes into the smaller lower section then up to the buzzing purple dot in the middle of the end of the upper section making various turns to look around. At 68s he exits the left side room towards the rectangle and moves in a clockwise circle.*

# Echolocation Game

License: GPL v3 (https://www.gnu.org/licenses/gpl-3.0)

Experimental prototype for a 2D FDTD echolocation simulation game using CUDA, SDL2, OpenGL, and GLEW. 

Navigate the room without eyes just by sound clicks. Featuring a dual view (static and first-person), sliding collisions, and a virtual cane (cane was bugged and is currently removed).

The sound wave simulation algorithm employs a Finite-difference time-domain (FDTD) to solve the 2D wave equation on a grid, leveraging CUDA to parallelize pressure updates and reflections, simulating wave propagation at 343 m/s with a spatial resolution of 0.01 meters and a 48 kHz audio capture rate.

The code was written with Grok-3 mainly, but also ChatGPT-o1. They were unable to optimize speed even further. GPU utilization on 3090 is fairly high.

The current result seems fairly convincing to me. In the brief period I had time to actually close my eyes and try, I noticed that it triggers my very limited ability of echolocation somewhat (I can see very faint shades, rotations and movements of objects with closed eyes). But from just 10 minutes of playing, depending on the hour of the day, I scored from awful to sort of ok in being able to navigate the maps.

This prototype was mainly written for baseline comparison purposes, to determine how well echolocation can work with highly accurate simulated sound waves.

Please beware that there is no scientifically established fool-proof method to simulate sound wave reflections accurately. And more can be done, such as multi-spectral reflection dynamics, which I found to be overkill for a prototype (I tried it and each frequency doubles compute, maxing out my 3090 with 3 frequencies). Testing was mainly done on the basis of what impression it made to me while listening, and how well it triggered my own limited ability of echolocation. I was mainly trusting in Grok-3's incredible levels of proficiency with the subject and algorithms used, without any strict formal and scientific evaluation.


## Vison for the future

If this program turns out to trigger people easily and consistently who can already echolocate in real life, more simplified an less computationally extreme simulation methods could be tested.

Once a room-sized 3D simulation is computationally possible (which it isn't with the current method), it opens all kinds of possibilities for blind people to see structures with sound, not only limited to games, but possibly also embossed text, UI interfaces, websites etc. for crude naviation. The accuracy to which echolocation can work in people given idealized conditions has yet to be studied though.

It could also be used as a training tool for people to learn to echolocate, since it provides a clean simple environment and highly idealized conditions.

Reducing the accuracy of the simulation unfortunately results in poor sound quality very fast. 

There are lots of other methods for aurealization, some faster some slower. But next to none of them are based on actual sound wave simulations. And those that do that I have seen, require huge 100GB pre-computed maps for simple rooms/buildings to work at all in 3D space, which heavily limits the usability of those methods to blind people. When I tested those methods, from what I can tell from my own very limited ability of echolocation, it either triggered it sometimes even remarkably, but then it seemed all jumbled up and distorted like a broken 3D mesh. Or it didn't trigger anything at all, indicating that spatial information in sound had been lost somehow. Since echolocation in humans is an unstudied phenomenon, it would probably be very unwise to start out with methods that make a lot of assumptions and optimizations to cut corners, which could be detrimental to how echolocation works.

Using Grok-3 I also managed to switch the simulation from FDTD (wave based) to a much faster non-FDTD ray based method using RIR pre-computation (not on Github). However it struggled to implement this, and the quality of sound was shockingly crude and abysimal by comparison to FDTD wave simulation. From what I understand from research papers, this is to be expected, and dozens of clever mathematical tricks that I don't really understand, and neither Grok-3 could come up with easily, would be necessary to somewhat improve that. Perhaps this project can be easily used as a template for Grok-4 and next-generation AI to rapidly advance the vision of more powerful echolocation simulations.


## Features
- Levels are loaded via SVG (drawing.svg or first command parameter)
- Echolocation: Pulses inside reasonable accurate physics simulation emitting from funnel forward from the player
- Controls: WASD/arrows (move), Q/E (strafe), mouse (cane).
- Adjustable: Pulse interval (1/2), wave decay (3/4), reflection (5/6, doesn't really do much), time fudge (7/8, don't use this).
- Player: 16cm mic distance, funnel apex at (0,0), pulse 1 unit below, mics 2 units below apex.

## Bugs

- Grok-3 messed up the sliding on walls, it doesn't work consistently, and fixing it became too much of a nightmare.
- adjustibles parameters mapped on number keys 1-8 can bug out if you increase values to extremes
- the "O", "I", or "W" elements were meant to play sounds for "dead end", "item" and "win game", but this is bugged so they only buzz ("win game" has much higher pitch)

## Creating levels

![Image](https://github.com/user-attachments/assets/e497c606-967c-4250-b281-084434ca7ebf)

*The image contains a side-by-side view of the level evilmaze.svg in Inkscape SVG editor and as rendered by the game.* 

Use Inkscape and save as plain SVG. The page dimensions must be in pixels and match SIZE (500) specified in .cu file.

- a Text element with with "O", "I", or "W" is interepreted as a pink, yellow or cyan buzzing emitter (not wave simulated)
- only opaque square shapes and circles were tested, star shapes, lines and such do not work
- don't rotate shapes it doesn't work

Load levels like so:
    ./echolocation_game level.svg

## Hardware

The exact hardware requirements for the default 500x500cm room are unknown, but approximately half that of a 3090. Memory usage is below 1GB and utilization 65% with 3090.

## Running on Windows

There is an untested Windows build available with CUDA 12.8 (=your CUDA driver must be >=12.8). You can download it in the [Releases](https://github.com/ballerburg9005/echolocation-game/releases) section on the right.

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
* Install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
* Install [Microsoft Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and pick "individual component" -> "MSVC v142 - VS 2019 C++ x64/x86 build tools"

BEWARE: with this combo is an error currently. It works with CUDA 12.8 and recent Visual Studio Code Community edititon if you pick option "Desktop Environment with C++" during install.

* **You must use the MINGW64 shell of MSYS2!**
```
MSVC_PATH=$(ls -d "/c/Program Files (x86)/Microsoft Visual Studio/"*/BuildTools/VC/Tools/MSVC/*/bin/HostX64/x64 | head -n 1)
CUDA_PATH=$(ls -d "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v"*/bin | head -n 1)
export PATH="$MSVC_PATH:$CUDA_PATH:/mingw64/bin:$PATH"
export CC=/mingw64/bin/gcc
export CXX=/mingw64/bin/g++
pacman -Syu
pacman -S mingw-w64-x86_64-toolchain mingw-w64-x86_64-SDL2 mingw-w64-x86_64-glew git wget unzip base-devel mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake
wget https://github.com/libsdl-org/SDL/releases/download/release-2.32.0/SDL2-devel-2.32.0-VC.zip
unzip SDL2-devel-2.32.0-VC.zip
mkdir -p ~/SDL2-2.32.0/include/SDL2
ln -s ~/SDL2-2.32.0/include/*.h ~/SDL2-2.32.0/include/SDL2/
wget https://github.com/nigels-com/glew/releases/download/glew-2.0.0/glew-2.0.0-win32.zip
unzip glew-2.0.0-win32.zip
wget https://github.com/leethomason/tinyxml2/archive/refs/tags/10.0.0.zip
unzip 10.0.0.zip
cd tinyxml2-10.0.0
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -DCMAKE_INSTALL_PREFIX=$HOME/tinyxml2-msvc -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ..
cmake --build . --config Release
cmake --install .
cd $HOME
git clone https://github.com/ballerburg9005/echolocation-game.git
cd echolocation-game
nvcc -o echolocation_game echolocation_game.cu -I$HOME/glew-2.0.0/include -L$HOME/glew-2.0.0/lib/Release/x64 -I$HOME/SDL2-2.32.0/include -L$HOME/SDL2-2.32.0/lib/x64 -I$HOME/tinyxml2-msvc/include -L$HOME/tinyxml2-msvc/lib -lSDL2main -lSDL2 -lopengl32 -lglew32 -ltinyxml2 -allow-unsupported-compiler -Xlinker /SUBSYSTEM:CONSOLE -diag-suppress 20012              


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
