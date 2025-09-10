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
        weights: Vec<f64>,
        half: i32,
    }

    impl Filter for FrameBlender {
        const NAME: &'static str = "Blend";
        const ARGS: &'static str = "clip:vnode;weights:float[]:opt;";
        const RETURNTYPE: &'static str = "clip:vnode;";
        const MODE: FilterMode = FilterMode::Parallel;

        fn from_args(args: &Map, _core: &CoreRef) -> Result<Self, String> {
            let input_node = args.get_node("clip")?;

            let mut weights = args.get_float_array("weights").unwrap();

            if weights.len() % 2 == 0 {
                return Err("Weights array must have an odd number of elements".to_string());
            }

            let half = (weights.len() as f32 / 2.0).round() as i32;

            let mut total_weights = 0.0;
            for i in 0..weights.len() {
                total_weights += weights[i];
            }

            for i in 0..weights.len() {
                weights[i] /= total_weights;
            }
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
            for _ in 0..=self.weights.len() {
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
                    self.frame_blend::<u16>(&source_frames, &mut dst, plane);
                } else if vf.bytes_per_sample == 1 {
                    self.frame_blend::<u8>(&source_frames, &mut dst, plane);
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
            T: TryFrom<i32> + Copy + Into<f64>,
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
                    let mut acc = 0.0;
                    for i in 0..num_srcs {
                        unsafe {
                            let val: f64 = (*frame_data_ptrs[i].wrapping_add(w)).into();
                            acc += val * self.weights[i];
                        }
                    }
                    let actual_acc = (acc as i32).clamp(0, max_val);
                    unsafe {
                        dst_ptr
                            .wrapping_add(w)
                            .write(actual_acc.try_into().unwrap());
                    }
                }
                for i in 0..num_srcs {
                    frame_data_ptrs[i] = frame_data_ptrs[i].wrapping_add(stride as usize);
                }
                dst_ptr = dst_ptr.wrapping_add(stride as usize);
            }
        }
    }

    rustsynth::register_filters!(FrameBlender);
}
