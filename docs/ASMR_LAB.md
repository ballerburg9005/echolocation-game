# ASMR experiment — physically rasterized SVG heads in real 2D FDTD

**Separate branch:** `experiment/asmr-fdtd-wav-path-lab`  
**No changes to `main` or the original arcade mode.**

## Build and run

The input recording is always **`~/test.mp3`**. No command-line audio file
parameter is needed; the system `libmpg123` decoder feeds the moving
**FDTD mouth emitter**, not the speakers directly.

```bash
git fetch origin
git switch experiment/asmr-fdtd-wav-path-lab
git pull --ff-only
sudo apt-get install -y libmpg123-dev libsdl2-dev librsvg2-dev libcairo2-dev pkg-config
make test
make asmr
./build/asmr_room
```

To supply or replace your private input: copy it to `~/test.mp3` and press
**L** to reload. The recording need not be committed to GitHub.

Both head drawings have physically specified **millimetre** viewBoxes and
are rendered with librsvg/Cairo at sub-millimetre spatial detail, including
your original bezier curves, elliptic arcs and SVG transforms. The program
loads the user's copy from `~/` when present, otherwise the copies
checked into `asmr_shapes/`.

### Head geometries and eardrums

**The default listener is YOUR `listener_v2.svg`** — both mirrored halves
and the large pinna openings, rather than the old simple round head.
Its original **185.348 × 136.859 mm** viewBox maps to exactly
**0.185348 × 0.136859 m** in the FDTD field, centred at (1m,1m).
The two authored half-paths are stitched only across their central
straight seam. Their contours are static sound-reflecting SOLID cells.

The experiment uses your **8 cm child eardrum-to-eardrum** assumption:
microphone positions are **(0.960 m, 1.003 m)** and
**(1.040 m, 1.003 m)**. This is a user-specified model parameter,
not a claim that all children's anatomy has that spacing. The wide
pinna openings lead to approximately **8.4 mm-wide fluid canals**
carved from the interior ear recess toward the microphone locations
where the SVG did not originally leave a connected air path. This is
an explicit *acoustic discretization adaptation*; the drawn silhouette
itself is not altered. At 1.5 mm offline cells the channel occupies
around 5–6 cells across; at coarse live resolution its response is
less reliable. When an ear aperture cannot be sampled from actual
fluid, the program refuses that grid rather than reading solid=0
pressure.

Press **H** in live mode to cycle the alternative geometries:
`listener_v2.svg` (default), `listener_with_ear_channel.svg`, and
`listener.svg`. The previous simple listener uses the experimental
8 cm + 0.8×2 cm ear-canal allowance, giving **9.6 cm outer-recess mic
spacing**. Both channel-type modes use the 8 cm eardrum target.

The moving artist is YOUR **`speaker.svg`** (134.483 × 130.034 mm),
rotated so its recessed mouth always faces the listener's centre.
The mouth lies about **5 cm** ahead of the artist centre, physically
inside the open front recess. It emits MP3 pressure in the same solver.
The speaker silhouette additionally creates **genuine moving
fractional impedance scattering** sampled from the uploaded SVG
alpha texture in the CUDA stencil. It is *not* a moving hard mask:
pressure is preserved during motion to avoid numerical whips.
This is an approximation to a moving solid boundary, not a complete
fluid–structure or rigid-body acoustic calculation.

The mouse still maps 1:1 to the full room, subject to listener,
speaker and outside-wall collision clearance. A physical
**25 cm ruler** appears at the bottom of the view.

### Frequency and full-room physical scale

| Property | Default live | Offline 5/10/20 s render |
|---|---:|---:|
| Physical room | 2.0 × 2.0 m | 2.0 × 2.0 m |
| Grid cell | **3.5 mm** | **1.5 mm** |
| Acoustic cells per side | ~571 | ~1333 |
| Wave steps/s (CFL) | ~139k | ~323k |
| Ten-cells/wavelength reference | ~9.8 kHz | ~22.9 kHz |
| Source low-pass | **8.5 kHz** | **19 kHz** |
| Sound capture | two FDTD microphones | same two microphones |

Press **Q** to cycle 3.5 / 3 / 2.5 / 5 / 8 / 10 mm live grid
settings, with corresponding input passbands. Offline stays at 1.5 mm,
independently of the preview resolution; it may take substantially
longer than realtime to render even a short clip.

The room dimensions are **not 50 cm**. Every five seconds the terminal
prints `[METRES]` showing source position in metres, actual left/right
source-to-microphone distance, predicted flight time using
343 m/s, exact 2m room side, dx and dt. The 2D pressure field
decays differently from 3D spherical waves — that can make distance
changes feel weaker, even when the grid scale is correct. We have
not added artificial 3D distance gain or HRTF processing.

### Controls

| Control | Action |
|---|---|
| Mouse | Position the artist absolutely in the 2 × 2 m room |
| M | Toggle mouse versus WASD/arrow movement |
| V | Toggle the **actual CUDA pressure-wave visualization** |
| 9 / 0 | Decrease/increase wave exposure, not the physical pressure |
| H | Cycle the three *real FDTD* listener SVG shapes, default V2 |
| Q | Cycle real-time acoustic grid resolution |
| F5 / 5 | Render and then **loop exactly 5 s** |
| F6 / 1 | Render and then **loop exactly 10 s** |
| F7 / 2 | Render and then **loop exactly 20 s** |
| P | Restart the most recently completed replay loop |
| **Space** | Stop playback and return to live movement/recording |
| T | Toggle live FDTD audio preview |
| L | Reload `~/test.mp3` |
| R | Reset live wave field and path history |
| Esc | Quit |

### Exact-duration replay and visualization

The input path is recorded in wall-clock seconds, with
timestamps and artist positions. Pressing a render button selects
the last 5/10/20 seconds available, then runs the offline **same
pressure-stencil FDTD** in fixed acoustic time steps. The stereo
resampler's output is **trimmed or padded to exactly N × 48,000
interleaved stereo frames**, where N is 5, 10 or 20. This corrects
the old discrepancy where a graph-batch overrun or acoustic output
tail could make the requested length unclear. If the path has less
history than requested, the artist holds its last captured position
for the remainder.

The complete rendered stereo buffer is saved as
`asmr_last_05s.wav` / `asmr_last_10s.wav` /
`asmr_last_20s.wav`, with the companion timestamped
`asmr_last_XXs_path.csv`. These are **outputs only**: MP3 remains
the default input.

Playback begins only after the full offline calculation finishes.
The next loop copy is queued into SDL **before** the current copy
runs out; audio playback does not wait for a CUDA kernel. The
visible artist SVG follows the recorded path again, the bottom
playhead counts each exact loop, and the title shows elapsed/
total seconds. Press **Space** to leave looping playback.

The cyan/magenta waves are the exact CUDA pressure field read from
the GPU, not a visualization-only oscillator. Solid listener head
walls are drawn as hard reflecting geometry. The 2D shader displays
raw signed pressure with adjustable exposure.

## Acoustic limitations and validation

- This is **genuine two-dimensional FDTD**, not HRTF rendering,
  generic reverb, direct stereo panning or ray tracing. The two
  microphones sample the propagated field on opposite sides of the
  SVG head. CPU regressions separately test two-ear arrival order.
- The v2 ear canals and their minimum width are *acoustic
  approximations*, not anatomically measured 3D canals. Pinna-shaped
  2D outlines do not reproduce real vertical front/back HRTFs.
- Moving-speaker scattering uses fractional soft impedance instead
  of an energy-conserving moving boundary; no hard pressure snaps.
  This preserves source motion without pretending to solve 3D
  fluid–structure coupling.
- Outer walls use a lower broad-band reflection coefficient for
  rubber-like attenuation; no measured rubber frequency response.
- The GPU load must be benchmarked in actual runtime, not inferred
  from its board's wattage. Larger 1.5mm offline grids require much
  more work even though completed audio plays at normal speed.
- Use low headphone gain during initial tests. Numerical code
  compilation and offline CPU geometry checks cannot verify physical
  loudness or your subjective ASMR response.

The ASMR hypothesis paper motivates exploring proximity and sound
movement but **does not establish** that any particular motion will
induce ASMR:

https://doi.org/10.3389/fnbeh.2025.1688172
