use rustsynth::{
    core::CoreRef,
    filter::{traits::Filter, FilterDependency, FilterMode, RequestPattern},
    frame::{Frame, FrameContext},
    map::Map,
    node::Node,
};
use rustsynth_derive::vapoursynth_plugin;
use std::cmp::max;
use std::fmt::Debug;
use std::arch::x86_64::*;

#[inline]
fn is_avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[vapoursynth_plugin]
mod plugin {
    use rustsynth::{ffi, plugin::PluginConfigFlags, MakeVersion};
    use rustsynth_derive::vapoursynth_filter;

    const NAMESPACE: &'static str = "frameblenderrs";
    const ID: &'static str = "nz.anima.frameblender";
    const NAME: &'static str = "FrameBlender";
    const PLUGIN_VER: i32 = MakeVersion!(1, 0);
    const API_VER: i32 = ffi::VAPOURSYNTH_API_VERSION;
    const FLAGS: i32 = PluginConfigFlags::NONE.bits();

    #[vapoursynth_filter(video)]
    struct FrameBlender {
        input_node: Node,
        weights: Vec<i32>,
        half: i32,
    }

    impl Filter for FrameBlender {
        const NAME: &'static str = "Blend";
        const ARGS: &'static str = "clip:vnode;weights:float[]:opt;";
        const RETURNTYPE: &'static str = "clip:vnode;";
        const MODE: FilterMode = FilterMode::Parallel;

        fn from_args(args: &Map, _core: &CoreRef) -> Result<Self, String> {
            let input_node = args.get_node("clip")?;

            let weights_f64 = args.get_float_array("weights").unwrap();

            if weights_f64.len() % 2 == 0 {
                return Err("Weights array must have an odd number of elements".to_string());
            }

            let half = (weights_f64.len() as f32 / 2.0).round() as i32;

            // Normalize weights and convert to fixed-point integers (16.16 format)
            let total_weight: f64 = weights_f64.iter().sum();
            let weights: Vec<i32> = weights_f64
                .iter()
                .map(|&w| ((w / total_weight) * 65536.0) as i32)
                .collect();
            Ok(Self {
                input_node,
                weights,
                half,
            })
        }

        // Remove get_video_info - use default (same as input)

        fn get_dependencies(&self) -> Vec<FilterDependency> {
            vec![FilterDependency {
                source: self.input_node.clone(),
                request_pattern: RequestPattern::General,
            }]
        }

        fn request_input_frames(&self, n: i32, frame_ctx: FrameContext) {
            let clamp = n > i32::MAX - 1 - self.half;
            let lastframe = if clamp { i32::MAX - 1 } else { n + self.half };
            // request all the frames we'll need
            for i in max(0, n - self.half)..=lastframe {
                self.input_node.request_frame_filter(i, &frame_ctx);
            }
        }

        fn process_frame<'core>(
            &mut self,
            n: i32,
            _frame_data: &[u8; 4],
            frame_ctx: FrameContext,
            core: CoreRef<'core>,
        ) -> Result<Frame<'core>, String> {
            let mut frame_num = n - self.half;
            let mut source_frames = Vec::new();
            for _ in 0..self.weights.len() {
                let frame = self
                    .input_node
                    .get_frame_filter(max(0, frame_num), &frame_ctx)
                    .unwrap();
                source_frames.push(frame);
                if frame_num < i32::MAX - 1 {
                    frame_num += 1;
                }
            }

            let src = &source_frames[0];
            let vf = src.get_video_format().unwrap();
            let height = src.get_height(0);
            let width = src.get_width(0);
            let mut dst = Frame::new_video_frame(&core, width, height, &vf, Some(&src));

            let num_planes = vf.num_planes;
            for plane in 0..num_planes {
                if vf.bytes_per_sample == 2 {
                    if is_avx2_available() {
                        unsafe { self.frame_blend_u16_avx2(&source_frames, &mut dst, plane); }
                    } else {
                        self.frame_blend::<u16>(&source_frames, &mut dst, plane);
                    }
                } else if vf.bytes_per_sample == 1 {
                    if is_avx2_available() {
                        unsafe { self.frame_blend_u8_avx2(&source_frames, &mut dst, plane); }
                    } else {
                        self.frame_blend::<u8>(&source_frames, &mut dst, plane);
                    }
                } else {
                    return Err("Unsupported bytes per sample".to_string());
                }
            }
            Ok(dst)
        }
    }

    impl FrameBlender {
        fn frame_blend<T>(&self, srcs: &Vec<Frame>, dst: &mut Frame, plane: i32)
        where
            T: TryFrom<i32> + Copy + Into<i32>,
            <T as TryFrom<i32>>::Error: Debug,
        {
            let height = dst.get_height(plane);
            let stride = dst.get_stride(plane) / size_of::<T>() as isize;
            let width = dst.get_width(plane);
            let num_srcs = self.weights.len();

            let mut frame_data_ptrs = Vec::new();
            for frame in srcs {
                let data = frame.get_read_ptr(plane) as *const T;
                frame_data_ptrs.push(data);
            }

            let mut dst_ptr = dst.get_write_ptr(plane) as *mut T;

            let max_val = (1 << (size_of::<T>() * 8)) - 1;
            for _h in 0..height {
                for w in 0..width as usize {
                    let mut acc = 0i64;
                    for i in 0..num_srcs {
                        unsafe {
                            let val = (*frame_data_ptrs.get_unchecked(i).wrapping_add(w)).into();
                            acc += (val * *self.weights.get_unchecked(i)) as i64;
                        }
                    }
                    let actual_acc = ((acc >> 16) as i32).clamp(0, max_val);
                    unsafe {
                        dst_ptr
                            .wrapping_add(w)
                            .write(actual_acc.try_into().unwrap());
                    }
                }
                for i in 0..num_srcs {
                    unsafe { frame_data_ptrs[i] = frame_data_ptrs.get_unchecked(i).wrapping_add(stride as usize) };
                }
                dst_ptr = dst_ptr.wrapping_add(stride as usize);
            }
        }

        #[target_feature(enable = "avx2")]
        unsafe fn frame_blend_u8_avx2(&self, srcs: &Vec<Frame>, dst: &mut Frame, plane: i32) {
            let height = dst.get_height(plane);
            let stride = dst.get_stride(plane);
            let width = dst.get_width(plane);
            let num_srcs = self.weights.len();

            let mut frame_data_ptrs = Vec::new();
            for frame in srcs {
                let data = frame.get_read_ptr(plane);
                frame_data_ptrs.push(data);
            }

            let mut dst_ptr = dst.get_write_ptr(plane);

            for _h in 0..height {
                let mut w = 0;
                
                // Process 32 pixels at a time with AVX2
                while w + 32 <= width as usize {
                    
                    let mut acc_0 = _mm256_setzero_si256();
                    let mut acc_1 = _mm256_setzero_si256();
                    let mut acc_2 = _mm256_setzero_si256();
                    let mut acc_3 = _mm256_setzero_si256();
                    
                    for i in 0..num_srcs {
                        let src_ptr = frame_data_ptrs[i].wrapping_add(w);
                        
                        // Load 32 u8 pixels as a 256-bit register
                        let pixels_256 = _mm256_loadu_si256(src_ptr as *const __m256i);
                        
                        // Split into low and high 128-bit parts
                        let pixels_lo = _mm256_extracti128_si256(pixels_256, 0);
                        let pixels_hi = _mm256_extracti128_si256(pixels_256, 1);
                        
                        // Convert u8 to u16 for both halves
                        let pixels_lo_u16 = _mm256_cvtepu8_epi16(pixels_lo);
                        let pixels_hi_u16 = _mm256_cvtepu8_epi16(pixels_hi);
                        
                        // Split u16 to u32 for multiplication
                        let pixels_0 = _mm256_unpacklo_epi16(pixels_lo_u16, _mm256_setzero_si256());
                        let pixels_1 = _mm256_unpackhi_epi16(pixels_lo_u16, _mm256_setzero_si256());
                        let pixels_2 = _mm256_unpacklo_epi16(pixels_hi_u16, _mm256_setzero_si256());
                        let pixels_3 = _mm256_unpackhi_epi16(pixels_hi_u16, _mm256_setzero_si256());
                        
                        let weight = _mm256_set1_epi32(self.weights[i]);
                        
                        // Multiply and accumulate in 32-bit
                        acc_0 = _mm256_add_epi32(acc_0, _mm256_mullo_epi32(pixels_0, weight));
                        acc_1 = _mm256_add_epi32(acc_1, _mm256_mullo_epi32(pixels_1, weight));
                        acc_2 = _mm256_add_epi32(acc_2, _mm256_mullo_epi32(pixels_2, weight));
                        acc_3 = _mm256_add_epi32(acc_3, _mm256_mullo_epi32(pixels_3, weight));
                    }
                    
                    // Shift right by 16 bits (divide by 65536)
                    acc_0 = _mm256_srai_epi32(acc_0, 16);
                    acc_1 = _mm256_srai_epi32(acc_1, 16);
                    acc_2 = _mm256_srai_epi32(acc_2, 16);
                    acc_3 = _mm256_srai_epi32(acc_3, 16);
                    
                    // Pack back to u16, then to u8
                    let result_lo_u16 = _mm256_packus_epi32(acc_0, acc_1);
                    let result_hi_u16 = _mm256_packus_epi32(acc_2, acc_3);
                    
                    // Pack u16 to u8 and combine
                    let result_lo_u8 = _mm256_extracti128_si256(result_lo_u16, 0);
                    let result_lo_u8_hi = _mm256_extracti128_si256(result_lo_u16, 1);
                    let result_hi_u8 = _mm256_extracti128_si256(result_hi_u16, 0);
                    let result_hi_u8_hi = _mm256_extracti128_si256(result_hi_u16, 1);
                    
                    let final_lo = _mm_packus_epi16(result_lo_u8, result_lo_u8_hi);
                    let final_hi = _mm_packus_epi16(result_hi_u8, result_hi_u8_hi);
                    
                    // Store 32 pixels
                    _mm_storeu_si128(dst_ptr.wrapping_add(w) as *mut __m128i, final_lo);
                    _mm_storeu_si128(dst_ptr.wrapping_add(w + 16) as *mut __m128i, final_hi);
                    
                    w += 32;
                }
                
                // Handle remaining pixels with scalar code
                while w < width as usize {
                    let mut acc = 0i64;
                    for i in 0..num_srcs {
                        let val = *frame_data_ptrs[i].wrapping_add(w);
                        acc += (val as i32 * self.weights[i]) as i64;
                    }
                    let result = ((acc >> 16) as i32).clamp(0, 255) as u8;
                    *dst_ptr.wrapping_add(w) = result;
                    w += 1;
                }
                
                // Move to next row
                for i in 0..num_srcs {
                    frame_data_ptrs[i] = frame_data_ptrs[i].wrapping_add(stride as usize);
                }
                dst_ptr = dst_ptr.wrapping_add(stride as usize);
            }
        }

        #[target_feature(enable = "avx2")]
        unsafe fn frame_blend_u16_avx2(&self, srcs: &Vec<Frame>, dst: &mut Frame, plane: i32) {
            let height = dst.get_height(plane);
            let stride = dst.get_stride(plane) / 2; // u16 stride
            let width = dst.get_width(plane);
            let num_srcs = self.weights.len();

            let mut frame_data_ptrs = Vec::with_capacity(num_srcs);
            for frame in srcs {
                let data = frame.get_read_ptr(plane) as *const u16;
                frame_data_ptrs.push(data);
            }

            let mut dst_ptr = dst.get_write_ptr(plane) as *mut u16;

            for _h in 0..height {
                let mut w = 0;
                
                // Process 16 pixels at a time with AVX2
                while w + 16 <= width as usize {
                    let mut acc_lo = _mm256_setzero_si256();
                    let mut acc_hi = _mm256_setzero_si256();
                    
                    for i in 0..num_srcs {
                        let src_ptr = frame_data_ptrs[i].wrapping_add(w);
                        let pixels = _mm256_loadu_si256(src_ptr as *const __m256i);
                        
                        // Unpack u16 to u32 for processing
                        let pixels_lo = _mm256_unpacklo_epi16(pixels, _mm256_setzero_si256());
                        let pixels_hi = _mm256_unpackhi_epi16(pixels, _mm256_setzero_si256());
                        
                        let weight = _mm256_set1_epi32(self.weights[i]);
                        
                        // Multiply and accumulate
                        acc_lo = _mm256_add_epi32(acc_lo, _mm256_mullo_epi32(pixels_lo, weight));
                        acc_hi = _mm256_add_epi32(acc_hi, _mm256_mullo_epi32(pixels_hi, weight));
                    }
                    
                    // Shift right by 16 bits (divide by 65536)
                    acc_lo = _mm256_srai_epi32(acc_lo, 16);
                    acc_hi = _mm256_srai_epi32(acc_hi, 16);
                    
                    // Pack back to u16
                    let result = _mm256_packus_epi32(acc_lo, acc_hi);
                    
                    _mm256_storeu_si256(dst_ptr.wrapping_add(w) as *mut __m256i, result);
                    
                    w += 16;
                }
                
                // Handle remaining pixels with scalar code
                while w < width as usize {
                    let mut acc = 0i64;
                    for i in 0..num_srcs {
                        let val = *frame_data_ptrs[i].wrapping_add(w);
                        acc += (val as i32 * self.weights[i]) as i64;
                    }
                    let result = ((acc >> 16) as i32).clamp(0, 65535) as u16;
                    *dst_ptr.wrapping_add(w) = result;
                    w += 1;
                }
                
                // Move to next row
                for i in 0..num_srcs {
                    frame_data_ptrs[i] = frame_data_ptrs[i].wrapping_add(stride as usize);
                }
                dst_ptr = dst_ptr.wrapping_add(stride as usize);
            }
        }
    }

    rustsynth::register_filters!(FrameBlender);
}
