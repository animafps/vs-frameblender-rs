# vs-frameblender-rs

Rust version of [couleurm/vs-frameblender](https://github.com/couleurm/vs-frameblender) using [rustsynth](https://github.com/animafps/rustsynth)

```txt
Benchmark 1: vspipe rs.vpy /dev/null
  Time (mean ± σ):      1.421 s ±  0.042 s    [User: 4.117 s, System: 0.930 s]
  Range (min … max):    1.368 s …  1.520 s    10 runs
 
Benchmark 2: vspipe cpp.vpy /dev/null
  Time (mean ± σ):      1.594 s ±  0.108 s    [User: 5.206 s, System: 1.057 s]
  Range (min … max):    1.422 s …  1.724 s    10 runs
 
Summary
  vspipe rs.vpy /dev/null ran
    1.12 ± 0.08 times faster than vspipe cpp.vpy /dev/null
```

## Usage

`frameblenderrs.Blend(clip: vnode, weights: float[]) -> clip: vnode`
