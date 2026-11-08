https://github.com/user-attachments/assets/57bfd1c9-6b8d-4146-8984-5ac6a208fbf6

*The video shows the game being used. There are two 500px square viewports rendered side-by-side. The impression of the graphics is very simple like in early computer games such as Pac-Man. The full scene on the left-side viewport can easily be described: there is a vertical very thick wall about 1/4th to the left with an opening about 1/3rd from the bottom. In the top left corner is a purple dot. The right hand room has 3 medium-sized circles lined up vertically around the center on the right side spaced rougly with similar distance but higher distance to the boundaries of the level. There is a slightly larger almost square rectangle in the upper half of the entire level almost center but slightly more downwards. The player is near the bottom left corner facing downwards. The player is composed of two green dots representing the virtual microphones, with a yellow hollow wedge in the middle that has a open end in the downward direction (i.e. two lines with one common origin point diverging roughly by 45 degrees). Inside that wedge is a red dot which is the sound pulse origin. The right-side viewport renders the same scene relative to the player's rotation fixed in the middle facing always upwards. The viewport is zoomed in considerably (4x) and there is as much spacing between the pixels, indicating that one pixel corresponds to the resolution of the FDTD wave simulation (1px=1cm). During the simulation sound waves flash up on the screen from the pulses now and then, but due to the speed of sound being much higher than frame rate, they are only weakly and momentarily visible. The borders of the level as well as the objects inside it and the wedge are yellow, indicating that they reflect sound waves. The background is black. From 0s to 14s the player orients himself and moves towards the front of the rectangle near the middle, continuing until 19s to move back and forth towards it, he then moves clockwise around it to face the upper circle from the left side and bumps into it. From 28s on he slaloms around the next circle while bumping into the upper and lower circle and and the nearby wall. From 35s he walks through the door and bumps into the left side of it at 41s, goes into the smaller lower section then up to the buzzing purple dot in the middle of the end of the upper section making various turns to look around. At 68s he exits the left side room towards the rectangle and moves in a clockwise circle.*

# Echolocation Game

## FDTD-only world acoustics and echo-first listening (experimental)

The original wedge was a makeshift impenetrable shield protecting the ears
from the direct click. Replacing it with a mere visual outline removed
movement-induced mask pressure artifacts, but also removed its acoustical
shield. This branch now includes two **optional A/B listening experiments**:

- **B — phased vs. isotropic FDTD click.** The default click uses two
  pressure sources separated by about 6cm: the rear element has opposite
  sign and is delayed by the sound transit time, interfering destructively
  behind the player. It DOES NOT turn the player into an impenetrable body,
  nor does it guarantee no direct sound at the ears. In a free-field CPU
  FDTD regression using the same stencil, backward receiver energy was
  reduced by about 98% relative to a single point source; that is a
  numerical bench test, NOT a perceptual calibration.
- **J — experimental echo-only headphone monitor (OFF by default).**
  Independently FDTD-simulates the same click in an empty room and
  subtracts that "incident" waveform from the receiver samples *before
  resampling*. The world pressure still contains ALL sound and reflects
  normally. This is a finite-difference incident/scattered field
  decomposition, not a physically impossible "sound reflects but never
  propagates until it hits something" shortcut. It is approximate when
  the player moves during an emitted pulse and adds CPU work, so use J
  to compare rather than assuming it is already more accurate.
- **N — disable/enable quiet FDTD world sounds.** Ghosts, generated candy
  shards, exits, and legacy SVG O/I/W landmarks emit distinct, gently
  modulated tones directly into the pressure field. Propagation, room
  occlusion and 2D reflections are therefore handled by FDTD.
  They are intentionally quiet but may still resonate in small rooms;
  toggle N when testing echo-only navigation. Collision taps and short
  game-rule reward sounds remain direct UI feedback, not FDTD.
- **Faceted acoustic objects:** ghosts are **16-gon reflectors** and
  candy shards are octagons, instead of smooth solid disks. This makes
  physical contours available to the grid but does not guarantee that
  facets will sound perceptually distinct at the current wavelength.
- **Collision and game-over bug fix:** enemy hits push the player away
  (with three seconds of protection) instead of instantly teleporting
  every ghost FDTD reflector back to spawn. After game over, enemies
  and remaining shards stay visible and acoustically present; **R**
  restarts the round.
- **2D wave gradient:** wave colors use logarithmic display-only
  exposure, not the old aggressive tanh saturation that made broad
  flat-color bands. **9/0** adjust *display* brightness without
  changing the simulated pressures or audio gain.

Both CUDA compilation and CPU numerical tests are useful, but none of this
replaces a headphone experiment on the intended NVIDIA GPU. Keep volume
low when testing J, B or the new in-field tones.

## Listening and camera corrections (following real-player feedback)

The latest iteration addresses issues observed on hardware in an earlier
version. These fixes are intentionally conservative and still need another
real-world listening check:

- **Moving/turning rattle:** the moving funnel outline is no longer stamped
  as a hard reflective wall into the FDTD grid; it is shown as a 2D visual
  overlay only. Receiver positions now interpolate over each audio batch.
  A new FDTD phased emitter attempts front/back directional emission, but
  the ear geometry and head scattering remain unvalidated.
- **3D mirror:** grid +Y now maps to GL -Z in mesh, camera and actors;
  A/left turning and D/right turning have consistent handedness.
- **3D circles/pillars:** ceiling and object tops no longer overlap at
  exactly the same depth (previously causing z-fighting lines); rounded
  actors use higher-detail spheres, smoother material shading and MSAA
  where supported.
- **Wall contact:** instead of repeated, identical 440/330Hz tones, find
  the actual wall cell touched along the collision circle. One short
  synthetic tap uses near/far-ear distance attenuation, relative ear delay,
  and a distinct rear sound. No periodic tone while holding into a wall.
  This is *HRTF-lite gameplay feedback*, not human binaural simulation.
- **No mysterious artificial whole-level beacon:** the older artificial
  O/I/W chimes have now been removed entirely. Instead the markers emit
  quiet FDTD tones, subject to numerical propagation and reflections
  (they can still be audible from far away). The optional direct F5/F6
  aids are OFF by default; F6 only operates within 35cm.

If you want to isolate the FDTD sounds, toggle off the arcade with **F4**
and leave F5/F6 off. Start headphone volume low. The project still needs
actual NVIDIA hardware A/B listening tests; compile/test success cannot
prove that the rattle is gone or the contact cues perceptually localize.

## Experimental improvements in this branch

This upgrade retains the original 2D CUDA FDTD concept and SVG levels, but
fixes important correctness and usability issues.

- **Physics:** stable 2D Courant timestep; rigid-wall (zero-gradient) boundary
  and a first-order impedance approximation for partial reflections, instead
  of sign-flipping the pressure inside solid cells.
- **Clock and audio:** simulation runs at its natural ~48.5k steps/second;
  microphone values are smoothly resampled to the SDL audio device rate.
  No arbitrary simulation-time fudge or aggressive queue resets.
- **Performance:** pressure, pulse and decay are fused, cutting four CUDA
  launches per time step to two. Unsafe synchronization on partial tiles
  is removed.
- **Sound:** a tapered 1.8 kHz/2 ms burst replaces the abrupt 500 Hz pulse;
  microphones use bilinear sampling and audio has a soft limiter.
- **Navigation:** substepped, test-covered wall sliding and collision
  prevents tunneling on slow frames.
- **Blind-play cues:** objects radiate their own deliberately quiet FDTD
  tones, with distinct frequencies and geometry; contact and short
  score/collection announcements are separate UI feedback.
- **SVG:** native circles, ellipses and rectangles, Inkscape elliptical
  arcs, basic fill/stroke handling and clipped rasterization.

## New: Ghost Circuit (experimental arcade mode)

The old experimental branch already contained a separate 3D rendering idea.
This branch recovers and modernizes the concept as a lightweight OpenGL
**first-person 2.5D view built from the same acoustically reflective grid**.

There is now a sound-driven arcade game: three wandering enemies
that chase nearby pings and act as dynamic CUDA wave reflectors, up to twelve
golden shards that also reflect waves, three lives, scoring, an unlockable
exit, collisions and victory sounds. Ghosts, shards and goal markers now
also radiate quiet **physical pressure-field tones**. Optional direct
non-FDTD proximity helpers are off by default. A new sample maze, **Ghost Circuit**, lives at
[echo_arena.svg](echo_arena.svg).

Launch the new level with:

    ./build/echolocation_game echo_arena.svg

Important new controls: **F** manual ping, **T** toggle automatic pulses
(stealth), **G** throw an FDTD sound decoy (9-second cooldown), **C** three-ray
virtual cane scan, **F2** swap 2D/3D view, **F3** immersive 3D,
**F4** arcade on/off, **F5** ghost heartbeat on/off,
**F6** guidance beacon on/off, **R** restart.

Enemies now have distinct tracker/sprinter/sentinel personalities, with
different speeds, hearing ranges and audio/visual identities. They hear
through reachable maze passages rather than a simple straight-line radius,
and pursue where the last ping occurred instead of knowing your live position.
A thrown decoy excites the same FDTD wave grid as a real click; the virtual
cane's measured obstacle taps are explicitly a separate synthetic audio aid.
The game finds a legal starting spot on each level and restarts there.

**New: sonic overdrive.** Every four echo shards grant a charge.
Press **H** to spend it: a physical FDTD player pulse goes out, ghosts
freeze for five seconds, and touching them during that period earns escalating
Pac-Man-style combo points. They temporarily disappear from the wave mask
until the power wears off. Freezing/capturing is an arcade rule, not
physical acoustics.
The original WASD/arrows/QE movement, Space visibility and numeric wave
parameter controls are retained. See [arcade guide](docs/ARCADE.md).

These improvements are **not proof of real-world echolocation fidelity**.
The 2D model has no HRTF, realistic 3D acoustics or external validation.
See [physics and validation](docs/VALIDATION.md). Begin at a low
headphone volume; this game is not a real-world navigation aid.

The CPU regression suite runs without NVIDIA hardware:

    for test in tests/test_*.cpp; do
      g++ -std=c++17 -O2 -Wall -Wextra -Werror "$test" -o /tmp/echo-test &&
        /tmp/echo-test || exit 1
    done

Full CUDA builds, audio output and usability still require GPU testing.


License: GPL v3 (https://www.gnu.org/licenses/gpl-3.0)

Experimental prototype for a 2D FDTD echolocation simulation game using CUDA, SDL2, OpenGL, and GLEW. 

Navigate the room by clicks, with a dual static/first-person view, sliding
collisions and a C-key three-ray virtual cane aid (the historical
mouse-driven cane remains removed).

The sound wave simulation algorithm employs a Finite-difference time-domain (FDTD) to solve the 2D wave equation on a grid, leveraging CUDA to parallelize pressure updates and reflections, simulating wave propagation at 343 m/s with a spatial resolution of 0.01 meters and a 48 kHz audio capture rate.

The code was written with Grok-3 mainly, but also ChatGPT-o1. They were unable to optimize speed even further. GPU utilization on 3090 is fairly high.

The current result seems fairly convincing to me. In the brief period I had time to actually close my eyes and try, I noticed that it triggers my very limited ability of echolocation somewhat (I can see very faint shades, rotations and movements of objects with closed eyes). But from just 10 minutes of playing, depending on the hour of the day, I scored from awful to sort of ok in being able to navigate the maps.

This prototype was mainly written for baseline comparison purposes, to determine how well echolocation can work with highly accurate simulated sound waves.

Please beware that there is no scientifically established fool-proof and computationally feasible method to simulate sound waves including reflections, decay, etc. accurately. And more can be done, such as multi-spectral reflection dynamics, which I found to be overkill for a prototype (I tried it and each frequency doubles compute, maxing out my 3090 with 3 frequencies). Testing was mainly done on the basis of what impression it made to me while listening, and how well it triggered my own limited ability of echolocation. I was mainly trusting in Grok-3's incredible levels of proficiency with the subject and algorithms used, without any strict formal and scientific evaluation.


## Vison for the future

If this program turns out to trigger people easily and consistently who can already echolocate in real life, more simplified an less computationally extreme simulation methods could be tested.

Once a room-sized 3D simulation is computationally possible (which it isn't with the current method), it opens all kinds of possibilities for blind people to see structures with sound, not only limited to games, but possibly also embossed text, UI interfaces, websites etc. for crude naviation. The accuracy to which echolocation can work in people given idealized conditions has yet to be studied though.

It could also be used as a training tool for people to learn to echolocate, since it provides a clean simple environment and highly idealized conditions.

Reducing the accuracy of the simulation unfortunately results in poor sound quality very fast. 

There are lots of other projects for aurealization, some faster some slower. But next to none of them are based on actual sound wave simulations. I have seen some that require huge 100GB pre-computed maps for simple rooms/buildings to work at all in 3D space, which heavily limits the usability of those methods to blind people. When I tested those methods, from what I can tell from my own very limited ability of echolocation, it either triggered it sometimes even remarkably, but then it seemed all jumbled up and distorted like a broken 3D mesh. Or it didn't trigger anything at all, indicating that spatial information in sound had been lost somehow. Since echolocation in humans is an unstudied phenomenon, it would probably be very unwise to start out with methods that make a lot of assumptions and optimizations to cut corners, which could be detrimental to how echolocation works.

Using Grok-3 I also managed to switch the simulation from FDTD (wave based) to a much faster non-FDTD ray based method using RIR pre-computation (not on Github). However it struggled to implement this, and the quality of sound was shockingly crude and abysimal by comparison to FDTD wave simulation. From what I understand from research papers, this is to be expected, and dozens of clever mathematical tricks that I don't really understand, and neither Grok-3 could come up with easily, would be necessary to somewhat improve that. Perhaps this project can be easily used as a template for Grok-4 and next-generation AI to rapidly advance the vision of more powerful echolocation simulations.


## Features, controls and known limitations

- Load a 500×500 pixel SVG level using a command such as
  ./echolocation_game drawing.svg.
- W/S or Up/Down: move. A/D or Left/Right: rotate. Q/E: strafe.
- Space: toggle visualization of reflective objects. Escape: quit.
- F manually generates a click; T enables/disables automatic pulses (stealth).
- G projects a sound decoy into a legal point ahead; C emits stereo left/center/right virtual cane taps.
- H spends a collected overdrive charge to freeze and catch ghosts for five seconds.
- F2 toggles right viewport 2D/3D; F3 toggles full-window first-person 3D.
- F4 toggles Ghost Circuit game mode; F5 nearby enemy heartbeat,
  F6 nearby target beacon. F5/F6 are OFF initially.
- B compares phased/isotropic FDTD player sources; N toggles world FDTD
  tones; J compares experimental direct-field subtraction (off by default).
- 9/0 changes only the 2D wave display exposure, not physics.
- R restarts all lives, shards, enemies, acoustic state and item pickups.
- Keys **1/2**: pulse interval; **3/4**: per-step wave decay;
  **5/6**: wall reflection; **7/8**: echolocation playback gain.
  Adjustable values are clamped.
- O markers: one low beep; I markers: two medium-pitched beeps (item);
  W markers: three high-pitched beeps (goal).
- Ghosts and shards are distinguishable in 3D, in 2D, and acoustically.
- Generated shard/exit guidance and threat heartbeats are nonphysical game
  audio cues and are optional. They are not part of the 2D wave simulation.
- The C-key virtual cane is a three-ray obstacle sonification aid, not a
  physical cane or acoustic scattering model. The old mouse-driven cane
  remains unimplemented. There is no speech tutorial, key remapping,
  or true 3D binaural HRTF rendering.
- Real-time output requires a sufficiently fast CUDA GPU. If a GPU cannot
  keep up, audio may underrun rather than silently using incorrect time.

## Creating levels

Use Inkscape and a 500×500 pixel document with a 1:1 viewBox.
The SVG vertical axis is flipped for simulation coordinates.

The loader recognizes untransformed rectangles, circles, ellipses, and
Inkscape elliptical-arc paths. It rasterizes basic fill/stroke geometry;
arbitrary paths, transforms, CSS inheritance and complex viewBox scaling
remain unsupported. A warning is logged for transforms.

A text element containing exactly O, I or W becomes a FDTD-emitting
acoustic landmark with a distinct low-level characteristic frequency.
The static sources remain in the simulated wave field until collected,
and can be found by listening to their waves and reflections.
Approaching to approximately 13 cm collects an item or activates the goal.
The earlier artificial proximity chirps have been removed; these sources
can in principle be heard beyond the old hard proximity cutoffs.

Load an example level:

    ./echolocation_game evilmaze.svg

## Hardware

The exact hardware requirements for the default 500x500cm room are unknown, but approximately half that of a 3090. Memory usage is below 1GB and utilization 65% with 3090.

## Running on Windows

There is an untested Windows build available with CUDA 12.8 (=your CUDA driver must be >=12.8). You can download it in the [Releases](https://github.com/ballerburg9005/echolocation-game/releases) section on the right.

## Build and Run

### Linux
```
sudo apt install build-essential libsdl2-dev libglew-dev libtinyxml2-dev libglm-dev libgl1-mesa-dev nvidia-cuda-toolkit
git clone https://github.com/ballerburg9005/echolocation-game.git
cd echolocation-game
git switch improve/physics-audio-accessibility-2026
make test
make build
./build/echolocation_game
```

### Windows (MSYS2)

**Note:** the original Windows instructions below are historical and have not been verified for this development branch. This branch currently has automated Linux CUDA compilation and CPU unit tests only.

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

See **Features, controls and known limitations** above. Nonvisual cues
are provided, but screen-reader guidance and user studies are still needed.

## License
GPLv3. See LICENSE.

## Contributing

Please include a reproducible numerical, collision, or SVG test for
applicable changes. CPU tests are in tests/ and run on GitHub Actions.
CUDA builds and listening tests remain essential for complete verification;
claims of acoustic fidelity require independent physical measurements.
