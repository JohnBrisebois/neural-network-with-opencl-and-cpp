// Using leaky relu because it produced the best outputs
inline float leaky_relu(float x) {
    return (x > 0.0f) ? x : 0.01f * x;
}

inline float dleaky_relu(float y) {
    return (y > 0.0f) ? 1.0f : 0.01f;
}

// Forward pass
__kernel void forward_batch(
    __global const float* prev_out,  // [B*prev_n]
    __global const float* weights,   // [curr_n*prev_n]
    __global const float* bias,      // [curr_n]
    __global float* curr_out,        // [B*curr_n]
    const int prev_n,
    const int curr_n,
    const int B)
{
    int gid = get_global_id(0);
    if (gid >= B * curr_n) return;

    int sample = gid / curr_n;
    int node   = gid % curr_n;

    float s = bias[node];
    const float* p_prev = prev_out + sample * prev_n;
    int wbase = node * prev_n;

    for (int k = 0; k < prev_n; ++k)
        s += p_prev[k] * weights[wbase + k];

    curr_out[sample * curr_n + node] = leaky_relu(s);
}

// Output delta
__kernel void output_delta(
    __global const float* out,      // [B*out_n]
    __global const float* target,   // [B*out_n]
    __global float* delta_out,      // [B*out_n]
    const int out_n,
    const int B)
{
    int gid = get_global_id(0);
    if (gid >= B * out_n) return;

    float o = out[gid];
    float t = target[gid];
    delta_out[gid] = (o - t) * dleaky_relu(o);
}

// Hidden layer delta
__kernel void hidden_delta(
    __global const float* hidden_out,   // [B*hidden_n]
    __global const float* next_weights, // [next_n*hidden_n]
    __global const float* delta_next,   // [B*next_n]
    __global float* delta_hidden,       // [B*hidden_n]
    const int hidden_n,
    const int next_n,
    const int B)
{
    int gid = get_global_id(0);
    if (gid >= B * hidden_n) return;

    int sample = gid / hidden_n;
    int h      = gid % hidden_n;

    float sum = 0.0f;
    for (int j = 0; j < next_n; ++j)
        sum += next_weights[j * hidden_n + h] * delta_next[sample * next_n + j];

    float hout = hidden_out[sample * hidden_n + h];
    delta_hidden[sample * hidden_n + h] = sum * dleaky_relu(hout);
}

// Weight gradients
__kernel void compute_gradients(
    __global const float* prev_out,   // [B*prev_n]
    __global const float* delta_curr, // [B*curr_n]
    __global float* grad,             // [curr_n*prev_n]
    const int prev_n,
    const int curr_n,
    const int B)
{
    int gid = get_global_id(0);
    if (gid >= curr_n * prev_n) return;

    int node = gid / prev_n;
    int k    = gid % prev_n;

    float s = 0.0f;
    for (int b = 0; b < B; ++b)
        s += prev_out[b * prev_n + k] * delta_curr[b * curr_n + node];

    grad[node * prev_n + k] = s;
}

// bias gradients
__kernel void compute_grad_bias(
    __global const float* delta_curr,  // [B*curr_n]
    __global float* grad_b,            // [curr_n]
    const int curr_n,
    const int B)
{
    int node = get_global_id(0);
    if (node >= curr_n) return;

    float s = 0.0f;
    for (int b = 0; b < B; ++b)
        s += delta_curr[b * curr_n + node];

    grad_b[node] = s;
}

// Apply weight updates
__kernel void apply_weights(
    __global float* weights,          // [curr_n*prev_n]
    __global const float* grad,       // [curr_n*prev_n]
    const float lr_over_B,
    const int total)
{
    int gid = get_global_id(0);
    if (gid >= total) return;
    weights[gid] -= lr_over_B * grad[gid];
}

// Apply bias updates
__kernel void apply_bias(
    __global float* bias,            // [curr_n]
    __global const float* grad_b,    // [curr_n]
    const float lr_over_B,
    const int total)
{
    int gid = get_global_id(0);
    if (gid >= total) return;
    bias[gid] -= lr_over_B * grad_b[gid];
}
