use crate::cnn::*;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

const GRID_DIM_X: u32 = 10;

const BLOCK_DIM_X: u32 = 24;
const BLOCK_DIM_Y: u32 = 24;
const BLOCK_DIM_Z: u32 = 1;

const OUT_BLOCK_DIM_X: u32 = 256;

const INTERMEDIATE_LAYER_DIM: usize = 4000;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.
pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>, // Conv kernels (10 5x5 kernels)
    output_layer: DeviceBox<OutputLayer>, // Weights (10 4000x1 matrices)
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        // Set up the context, load the module, and create a stream to run kernels in.
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let _context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

        Ok(Self {
            conv_layer: DeviceBox::new(&cnn.conv_layer)?,
            output_layer: DeviceBox::new(&cnn.output_layer)?,
            _context,
            module: Module::load_from_string(&CString::new(include_str!(
                "./../kernel/kernel.ptx"
            ))?)?,
            stream: Stream::new(StreamFlags::NON_BLOCKING, None)?,
        })
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        let mut hidden_layer = DeviceBuffer::from_slice(&[0f64; INTERMEDIATE_LAYER_DIM])?;
        let mut input_image = DeviceBox::new(input)?;
        // Create buffer for the final output vector.
        let mut host_output = OutputVec {
            0: [0f64; OUT_LAYER_SIZE],
        };
        let mut output = DeviceBox::new(&host_output)?;

        let module = &self.module;
        let stream = &self.stream;
        // Submit two pieces of work to the same stream.
        // The submissions of work are asynchronous in that the kernel call directly returns to the CPU.
        // However, the execution order of work on the same stream is sequential.
        // Hence, one synchronize() is required only.
        unsafe {
            launch!(module.cnn<<<GRID_DIM_X, (BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z), 0, stream>>>(
                input_image.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                hidden_layer.as_device_ptr())
            )?;

            launch!(module.output_layer<<<OUT_LAYER_SIZE as u32, OUT_BLOCK_DIM_X, 0, stream>>>(
                hidden_layer.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                output.as_device_ptr()
            ))?;
        }
        stream.synchronize()?;
        output.copy_to(&mut host_output)?;
        Ok(host_output)
    }
}
