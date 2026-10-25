#include "mmid-f32.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#    include "dequantize.cuh"

static bool mmid_f32_layout_supported(const ggml_tensor * dst,
                                      ggml_type           weight_type,
                                      std::size_t         block_bytes,
                                      int                 block_elements) {
    if (dst->op != GGML_OP_MUL_MAT_ID || ggml_get_op_params_i32(dst, 3) != GGML_PREC_F32) {
        return false;
    }
    const auto * a   = dst->src[0];
    const auto * b   = dst->src[1];
    const auto * ids = dst->src[2];
    if (a->type != weight_type || b->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32 || a->ne[0] <= 0 || a->ne[0] > 4096 || a->ne[0] % block_elements || a->ne[1] <= 0 ||
        a->ne[1] > 4096 || a->ne[2] <= 0 || a->ne[2] > 128 || a->ne[3] != 1 || b->ne[0] != a->ne[0] || b->ne[2] != 1 ||
        b->ne[3] != 1 || ids->ne[0] <= 0 || ids->ne[0] > 64 || ids->ne[0] > a->ne[2] || ids->ne[1] != 1 ||
        ids->ne[2] != 1 || ids->ne[3] != 1 || (b->ne[1] != 1 && b->ne[1] != ids->ne[0]) || dst->ne[0] != a->ne[1] ||
        dst->ne[1] != ids->ne[0] || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }
    const std::size_t row = std::size_t(a->ne[0] / block_elements) * block_bytes;
    return a->nb[0] == block_bytes && a->nb[1] == row && a->nb[2] >= std::size_t(a->ne[1]) * row &&
           a->nb[2] <= 8192 * row && a->nb[2] % block_bytes == 0 && b->nb[0] == sizeof(float) &&
           b->nb[1] >= std::size_t(b->ne[0]) * sizeof(float) && b->nb[1] <= 8192 * sizeof(float) &&
           b->nb[1] % sizeof(float) == 0 && ids->nb[0] == sizeof(int32_t) && dst->nb[0] == sizeof(float) &&
           dst->nb[1] >= std::size_t(dst->ne[0]) * sizeof(float) && dst->nb[1] <= 8192 * sizeof(float) &&
           dst->nb[1] % sizeof(float) == 0;
}

bool ggml_cuda_mul_mat_id_f32_supported(const ggml_tensor * dst, int cc) {
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || dst->op != GGML_OP_MUL_MAT_ID ||
        ggml_get_op_params_i32(dst, 3) != GGML_PREC_F32) {
        return false;
    }
    switch (dst->src[0]->type) {
        case GGML_TYPE_Q8_0:
            return mmid_f32_layout_supported(dst, GGML_TYPE_Q8_0, sizeof(block_q8_0), QK8_0);
        case GGML_TYPE_Q5_0:
            return mmid_f32_layout_supported(dst, GGML_TYPE_Q5_0, sizeof(block_q5_0), QK5_0);
        default:
            return false;
    }
}

static __global__ void dequantize_selected_q8(const char *    weights,
                                              const int32_t * ids,
                                              float *         out,
                                              int             elements,
                                              int             slot,
                                              int             experts,
                                              size_t          expert_stride) {
    const int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= elements) {
        return;
    }
    const int expert = ids[slot];
    if (expert < 0 || expert >= experts) {
        out[index] = out[index + 1] = __int_as_float(0x7fc00000);
        return;
    }
    const auto * blocks = reinterpret_cast<const block_q8_0 *>(weights + size_t(expert) * expert_stride);
    const auto & q      = blocks[index / QK8_0];
    // Supported row lengths are multiples of 32, so each pair stays in its block.
    const float  d      = __half2float(q.d);
    const float2 values =
        make_float2(__fmul_rn(d, float(q.qs[index % QK8_0])), __fmul_rn(d, float(q.qs[index % QK8_0 + 1])));
    reinterpret_cast<float2 *>(out)[index / 2] = values;
}

static void mul_mat_id_q8_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto *                a   = dst->src[0];
    const auto *                b   = dst->src[1];
    const auto *                ids = dst->src[2];
    const int                   k = int(a->ne[0]), m = int(a->ne[1]);
    const int                   elements = k * m;
    // Reuse one expert-sized buffer, retaining the reference's SGEMM dimensions.
    ggml_cuda_pool_alloc<float> expanded(ctx.pool(), elements);
    const float                 alpha = 1.0f, beta = 0.0f;
    auto                        stream = ctx.stream();
    auto                        handle = ctx.cublas_handle();
    for (int slot = 0; slot < ids->ne[0]; ++slot) {
        dequantize_selected_q8<<<(elements / 2 + 255) / 256, 256, 0, stream>>>(
            static_cast<const char *>(a->data), static_cast<const int32_t *>(ids->data), expanded.get(), elements, slot,
            int(a->ne[2]), a->nb[2]);
        CUDA_CHECK(cudaGetLastError());
        const auto * input =
            reinterpret_cast<const float *>(static_cast<const char *>(b->data) + size_t(slot % b->ne[1]) * b->nb[1]);
        auto * output = reinterpret_cast<float *>(static_cast<char *>(dst->data) + size_t(slot) * dst->nb[1]);
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, 1, k, &alpha, expanded.get(), k, input, k, &beta,
                                 output, m));
    }
}

static __global__ void dequantize_selected_q5(const char *    weights,
                                              const int32_t * ids,
                                              float *         expanded,
                                              int             elements,
                                              int             slot,
                                              int             experts,
                                              size_t          stride) {
    const int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= elements) {
        return;
    }
    const int    id          = ids[slot];
    const int    safe_id     = id >= 0 && id < experts ? id : 0;
    const auto * selected    = weights + size_t(safe_id) * stride;
    const int    quant_index = (index % QK5_0) / QR5_0;
    const int    block_start = index - index % QK5_0;
    // Keep the upstream converter's quant index, arithmetic and output layout.
    // Select the expert on device instead of copying its packed bytes first.
    float2       values;
    dequantize_q5_0(selected, index / QK5_0, quant_index, values);
    expanded[block_start + quant_index]             = values.x;
    expanded[block_start + quant_index + QK5_0 / 2] = values.y;
}

static __global__ void mask_invalid_q5(const int32_t * ids, int slot, int experts, float * output, int m) {
    const int id    = ids[slot];
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < m && (id < 0 || id >= experts)) {
        output[index] = __int_as_float(0x7fc00000);
    }
}

static void mul_mat_id_q5_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto *                a   = dst->src[0];
    const auto *                b   = dst->src[1];
    const auto *                ids = dst->src[2];
    const int                   k = int(a->ne[0]), m = int(a->ne[1]), elements = k * m;
    ggml_cuda_pool_alloc<float> expanded(ctx.pool(), elements);
    auto                        stream = ctx.stream();
    auto                        handle = ctx.cublas_handle();
    const float                 alpha = 1.0f, beta = 0.0f;
    for (int slot = 0; slot < ids->ne[0]; ++slot) {
        dequantize_selected_q5<<<(elements / 2 + 255) / 256, 256, 0, stream>>>(
            static_cast<const char *>(a->data), static_cast<const int32_t *>(ids->data), expanded.get(), elements, slot,
            int(a->ne[2]), a->nb[2]);
        CUDA_CHECK(cudaGetLastError());
        const auto * input =
            reinterpret_cast<const float *>(static_cast<const char *>(b->data) + size_t(slot % b->ne[1]) * b->nb[1]);
        auto * output = reinterpret_cast<float *>(static_cast<char *>(dst->data) + size_t(slot) * dst->nb[1]);
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, 1, k, &alpha, expanded.get(), k, input, k, &beta,
                                 output, m));
        mask_invalid_q5<<<(m + 255) / 256, 256, 0, stream>>>(static_cast<const int32_t *>(ids->data), slot,
                                                             int(a->ne[2]), output, m);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_cuda_mul_mat_id_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_Q8_0:
            mul_mat_id_q8_f32(ctx, dst);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_id_q5_f32(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported F32 expert weight type");
    }
}
#endif
