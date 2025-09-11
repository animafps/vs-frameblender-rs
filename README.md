# vs-frameblender-rs

Rust version of [couleurm/vs-frameblender](https://github.com/couleurm/vs-frameblender) using [rustsynth](https://github.com/animafps/rustsynth)

```txt
Benchmark 1: vspipe rs.vpy /dev/null
  Time (mean ± σ):      5.590 s ±  0.590 s    [User: 35.410 s, System: 1.905 s]
  Range (min … max):    4.495 s …  6.022 s    10 runs
 
Benchmark 2: vspipe cpp.vpy /dev/null
  Time (mean ± σ):      8.741 s ±  0.108 s    [User: 57.615 s, System: 1.905 s]
  Range (min … max):    8.609 s …  8.956 s    10 runs
 
Summary
  vspipe rs.vpy /dev/null ran
    1.56 ± 0.17 times faster than vspipe cpp.vpy /dev/null
```

## Usage

`frameblenderrs.Blend(clip: vnode, weights: float[]) -> clip: vnode`
