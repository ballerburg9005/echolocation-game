# ASMR spatial audio laboratory — real CUDA FDTD + MP3

**Experiment branch:** `experiment/asmr-fdtd-wav-path-lab`  
**Base:** the working arcade physics branch; the arcade binary remains separate.

This program uses **2D pressure-wave FDTD**. The source recording is
decoded from MP3 into mono PCM once, and the samples are added as
a moving pressure source **inside** the acoustic CUDA grid. Every pressure
time step uses the actual second-order finite-difference wave stencil
with a Courant-stable step, fixed listener-head obstacle, room reflections,
and two independently sampled virtual microphones. There is **no HRTF
convolution, ray tracing, sine-ring illustration, or artificial stereo pan**.

## Build and start

The program **always reads `~/test.mp3`** (from the OS user's HOME
directory), regardless of the working directory. There is no input
filename argument and no need to convert anything to WAV manually.

```bash
git fetch origin
git switch experiment/asmr-fdtd-wav-path-lab
git pull --ff-only
sudo apt-get install libmpg123-dev libsdl2-dev
make test
make asmr
./build/asmr_room
```

To place your file locally:

```bash
cp /path/to/your/recording.mp3 ~/test.mp3
./build/asmr_room
```

If you **want to commit and push the MP3** into this experiment branch
rather than keeping it on your own computer:

```bash
git switch experiment/asmr-fdtd-wav-path-lab
cp ~/test.mp3 test.mp3
git add test.mp3
git commit -m "Add ASMR source recording"
git push origin experiment/asmr-fdtd-wav-path-lab
```

**Note:** The running application still reads `~/test.mp3`, not the
copy in the repository. Only push recordings you have the rights to
share. You do **not** need to push your recording to run the program.

Press **L** to reload it after replacement. The native libmpg123
decoder handles MP3 and passes 16-bit PCM to SDL for 48 kHz float
mono conversion. The first 60 seconds are used, with a 20 ms crossfade
when looping. The source is independently low-pass-filtered for live
and offline resolutions, then uploaded once into CUDA memory.

## Controls

| Key or control | Action |
|---|---|
| Mouse | Absolute artist-head position in the 2m room, with solid head/wall collision |
| M | Toggle mouse mode; when off, use WASD or arrow keys |
| **V** | Show/hide **the actual GPU FDTD pressure field** (default visible) |
| **9 / 0** | Decrease/increase pressure visualization exposure, **no change to physics or audio** |
| **Q** | Cycle live FDTD resolution: **5 mm (default) → 8 mm → 10 mm**; use if the GPU misses live deadlines |
| F5 or 5 | Select last 5 seconds, offline render, save and play |
| F6 or 1 | Select last 10 seconds |
| F7 or 2 | Select last 20 seconds |
| P | Replay last fully processed file without running CUDA again |
| L | Reload `~/test.mp3` |
| T | Enable/disable FDTD live preview sound |
| R | Reset live pressure and recorded movement path |
| Esc | Quit |

The **cyan and magenta wavefronts** in the window are colored real values
from the SAME CUDA pressure field that drives the ear audio. The static
head is blue inside that grid. The movable orange artist head is
collision-checked, but is not a hard moving FDTD wall because stamping
hard pressure cells when turning can generate artificial gunshot-like
transients. The yellow spot is at the **actual bilinearly placed
pressure-emitting mouth**, which may lag sudden mouse motion by a few
milliseconds to prevent source-position zipper noise.

## Real acoustics, geometry and spatialization

The room is a 2m by 2m empty chamber with broad-band lossy
outer reflection R=0.32. A fixed 17cm-diameter listener head is
a **real hard FDTD mask**. The two virtual ear microphones are separated
by 21 cm and sample independently on opposite sides of this head.
The orange 17cm artist head always points its mouth toward the
listener's centre, and its moving MP3 samples enter the room field.

Pressure propagation around the stationary listener creates actual
**planar** acoustic shadow, reflection, diffraction and left/right
wave timing/pressure differences. This is still 2D acoustic physics,
not a calibrated 3D HRTF, individual pinnae, ear canals or elevation.

The real pressure field is displayed as colorful wavefronts; there is
no procedural graphics animation substituted for the physics.
Press **V** to compare with a static-map view and **9/0** if waves appear
too bright or faint. A pressure image snapshot adds a small GPU readback
cost; the solver itself is unchanged by visualization exposure.

## Why mouse movement previously sounded grainy

The previous live version repeatedly interpolated **the same old mouse
position to the same new mouse position once for EACH of 50+ CUDA
audio batches per frame**. So a single physical movement was
repeated many times, producing an unintended zipper/sawtooth
modulation that can sound like hiss. It also always applied
`0.42*tanh(2*p)` to the ear pressure, introducing artificial harmonics.

The new renderer maintains **one continuous pressure-source trajectory
across batch boundaries**. Source movement is interpolated *once* on the
physics time grid, with about 12ms smoothing and a conservative maximum
movement rate. This is a change to source geometry, **not an audio
denoising filter**; the FDTD scattering and high-frequency bandwidth
remain. The listener still hears raw left/right FDTD pressure with
only a single linear volume gain and a last-resort safety clip.

Audio from many small pressure batches is now consolidated into one
SDL queue operation per display update. The program prints
`[ASMR FDTD]` diagnostics with pressure steps, live grid resolution,
near-empty audio queues and safety-clipped sample counts. If you still
hear hiss, compare slow versus fast cursor motion and share those
lines. An MP3 can also contain its own compression artifacts.

## Quality and offline capture

| | Default live preview | High-resolution offline render |
|---|---:|---:|
| Cell size | **5 mm** | **2 mm** |
| Grid | 400×400 | 1000×1000 |
| Physical FDTD steps/s | ~97,000 | ~243,000 |
| 10-cells-per-wavelength frequency | ~6.86 kHz | ~17.15 kHz |
| Input low-pass before FDTD | **5.5 kHz** | **15 kHz** |
| Source file | `~/test.mp3` | the same MP3 |
| Audio to headphones | Real-time preview if the GPU keeps up | ONLY after complete rendering |

Pressing F5/F6/F7 freezes the path and processes the selected
5-, 10-, or 20-second window on the finer grid. This
runs the SAME 2D FDTD pressure solver at higher spatial and
temporal resolution and feeds the SAME decoded source PCM, but
with an up-to-15kHz passband to explore frequencies above 8kHz.

The completed stereo samples are stored in RAM, written as
`asmr_last_05s.wav` / `10s` / `20s`, then played through SDL without
render-time stuttering. The saved companion
`asmr_last_05s_path.csv` etc records artist/mouth trajectory and angles.
Generated WAV/CSV files are ignored by Git. They overwrite earlier
captures of the same duration; copy promising takes elsewhere.

The offline grid has **6.25 times as many cells** as the default
5mm live grid and 2.5 times as many steps per second: about
**15.6x** more pressure-cell updates per second of recorded audio.
For comparison with the old 10mm/2.5mm configuration, offline
work is also substantially higher. Power draw such as 120W on a
nominal 350W RTX 3090 does *not* imply that the GPU has a
particular percentage of spare realtime capacity. For a weaker
GPU, press **Q** to lower live resolution without reducing the
15kHz offline passband.

A reusable CUDA Graph advances 126 sequential acoustic pressure
steps before the CPU needs to launch another graph. Since 126 is
divisible by 3, the triple pressure-field buffers line up across
graph boundaries. The pressure field is not synthesized directly
at the headphones: only the two ear samples are read out.

**Headphone safety:** begin at low volume; no physical SPL
calibration is available. A safety amplitude limit protects against
unexpected numerical extremes, but should not be mistaken for a
measured acoustic level.

## ASMR exploration

The *Proximity Prediction Hypothesis* paper (Flockton et al., 2025)
discusses near-ear spatial signals and the importance of high-frequency
shadow cues above 8kHz. It is a theoretical account, not evidence
that any one movement is guaranteed to induce ASMR.

Useful comparisons include stationary near-ear playback,
slow approach–pause–withdrawal, moving from left to right
around the head while the mouth faces the listener, and alternating
near/far distances with a consistent source and headphone level.

https://doi.org/10.3389/fnbeh.2025.1688172

## Validation status

- CPU physics, geometry and path-timing tests run in GitHub Actions.
- CUDA CI compiles the executable, without running a GPU/audio session.
- Your own headphone testing is required for actual image quality,
  moving-source hiss, deadline performance and perceived head-shadow.
- **Not** a calibrated binaural HRTF/3D acoustics model or
  a clinical demonstration of ASMR.
