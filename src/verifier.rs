const EPSILON: f64 = 0.000001;

pub fn convolution_layer(input: &InputMatrix, conv_filters: &ConvLayer, outputs: &mut ConvOutput) {
    // Go through each convolution neuron
    for (filter, out) in conv_filters.0.iter().zip(outputs.0.iter_mut()) {
        // Divide the 100x100 input matrix into 5x5 regions. There are 20x20 such regions in the
        // matrix. Each convolution neuron does a dot product of its filter with each region, producing a
        // 20x20 output matrix of products
        for i in (0..INPUT_DIM).step_by(FILTER_DIM) {
            for j in (0..INPUT_DIM).step_by(FILTER_DIM) {
                // Dot product
                let prod: f64 = (0..FILTER_DIM)
                    .flat_map(move |x| {
                        (0..FILTER_DIM).map(move |y| input.0[i + x][j + y] * filter[x][y])
                    })
                    .sum();
                out[i / FILTER_DIM][j / FILTER_DIM] = prod;
            }
        }
    }
}

pub fn relu_layer(conv_out: &mut ConvOutput) {
    // Any value below 0 in the previous layer's output is changed to 0
    for matrix in conv_out.0.iter_mut() {
        for row in matrix {
            for val in row {
                if *val < 0.0 {
                    *val = 0.0;
                }
            }
        }
    }
}

pub fn output_layer(input: &ConvOutput, weights: &OutputLayer, output: &mut OutputVec) {
    // Go thru each output neuron
    for (weight, out) in weights.0.iter().zip(output.0.iter_mut()) {
        // Flatten the output of the previous layer into a 4000x1 vector, then dot product it with
        // the weight vector to produce a single value
        let flattened = input.0.iter().flat_map(|n| n.iter().flat_map(|r| r.iter()));
        let prod: f64 = flattened.zip(weight.iter()).map(|(a, b)| a * b).sum();
        *out = prod;
    }
}

// Verify the results from the first stage (up to relu).
// The CPU does the same computation and we compare the results computed by the gpu against that by the cpu.
pub fn verify_conv_results(
    input: &InputMatrix,
    hidden_layer: &[f64; INTERMEDIATE_LAYER_DIM],
    conv_output_buffer: &mut ConvOutput,
) -> bool {
    convolution_layer(input, &self.conv_layer_vec, conv_output);
    relu_layer(conv_output);
    let mut idx: usize = 0;
    for neuron in conv_output.0.iter() {
        for row in neuron.iter() {
            for col in row.iter() {
                if (*col - hidden_layer[idx]) > EPSILON {
                    println!("Results not close enough: {}, {}", hidden_layer[idx], *col);
                    return false;
                }
                idx += 1;
            }
        }
    }
    idx == INTERMEDIATE_LAYER_DIM
}

// Verify the final output.
pub fn verify_out_results(
    conv_output: &ConvOutput,
    weights: &OutputLayer,
    host_output: &OutputVec,
) -> bool {
    let mut host_output_cpu = OutputVec {
        0: [0f64; OUT_LAYER_SIZE],
    };
    output_layer(&conv_output, weights, &mut host_output_cpu);
    for i in 0..OUT_LAYER_SIZE {
        if (host_output_cpu.0[i] - host_output.0[i]).abs() > EPSILON {
            println!(
                "Results not close enough: {}, {}",
                host_output_cpu.0[i], host_output.0[i]
            );
        }
    }
    true
}
