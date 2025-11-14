/**
 * SYCL Kernels for MoE Routing Operations
 * Implements top-k expert selection, scatter/gather, and batched expert dispatch
 * Target: Intel Arc iGPU (Intel Core Ultra 9 185H)
 */

#ifdef ENABLE_SYCL
#include <sycl/sycl.hpp>
#include <cmath>
#include <algorithm>

namespace moe_kernels {

/**
 * Top-K expert selection kernel
 * Selects top-k experts based on router logits for each token
 * 
 * Input:
 *   - router_logits: [batch_size * seq_len, num_experts] - raw router scores
 *   - num_experts: number of available experts
 *   - k: number of experts to select per token
 * 
 * Output:
 *   - selected_experts: [batch_size * seq_len, k] - indices of selected experts
 *   - routing_weights: [batch_size * seq_len, k] - normalized routing weights
 */
class TopKExpertSelectionKernel {
public:
    void operator()(sycl::queue& q,
                   const float* router_logits,
                   int* selected_experts,
                   float* routing_weights,
                   int num_tokens,
                   int num_experts,
                   int k) {
        
        // Allocate device memory
        auto logits_dev = sycl::malloc_device<float>(num_tokens * num_experts, q);
        auto experts_dev = sycl::malloc_device<int>(num_tokens * k, q);
        auto weights_dev = sycl::malloc_device<float>(num_tokens * k, q);
        
        // Copy input to device
        q.memcpy(logits_dev, router_logits, num_tokens * num_experts * sizeof(float)).wait();
        
        // Launch kernel
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(num_tokens), [=](sycl::id<1> token_idx) {
                const int tid = token_idx[0];
                const float* token_logits = logits_dev + tid * num_experts;
                
                // Find top-k experts for this token
                // Using simple selection (for real use, consider radix select or sorting network)
                int top_k_indices[16];  // Assume max k=16
                float top_k_values[16];
                
                // Initialize with first k elements
                for (int i = 0; i < k; i++) {
                    top_k_indices[i] = i;
                    top_k_values[i] = token_logits[i];
                }
                
                // Find minimum in top-k
                auto find_min_idx = [&]() {
                    int min_idx = 0;
                    for (int i = 1; i < k; i++) {
                        if (top_k_values[i] < top_k_values[min_idx]) {
                            min_idx = i;
                        }
                    }
                    return min_idx;
                };
                
                // Process remaining experts
                for (int expert_idx = k; expert_idx < num_experts; expert_idx++) {
                    float logit = token_logits[expert_idx];
                    int min_idx = find_min_idx();
                    
                    if (logit > top_k_values[min_idx]) {
                        top_k_values[min_idx] = logit;
                        top_k_indices[min_idx] = expert_idx;
                    }
                }
                
                // Compute softmax over top-k logits
                float max_logit = top_k_values[0];
                for (int i = 1; i < k; i++) {
                    max_logit = sycl::fmax(max_logit, top_k_values[i]);
                }
                
                float sum_exp = 0.0f;
                for (int i = 0; i < k; i++) {
                    top_k_values[i] = sycl::exp(top_k_values[i] - max_logit);
                    sum_exp += top_k_values[i];
                }
                
                // Normalize
                for (int i = 0; i < k; i++) {
                    top_k_values[i] /= sum_exp;
                }
                
                // Write results
                for (int i = 0; i < k; i++) {
                    experts_dev[tid * k + i] = top_k_indices[i];
                    weights_dev[tid * k + i] = top_k_values[i];
                }
            });
        }).wait();
        
        // Copy results back
        q.memcpy(selected_experts, experts_dev, num_tokens * k * sizeof(int)).wait();
        q.memcpy(routing_weights, weights_dev, num_tokens * k * sizeof(float)).wait();
        
        // Free device memory
        sycl::free(logits_dev, q);
        sycl::free(experts_dev, q);
        sycl::free(weights_dev, q);
    }
};

/**
 * Token scatter kernel
 * Scatters tokens to their assigned experts and builds dispatch metadata
 * 
 * Input:
 *   - hidden_states: [num_tokens, hidden_dim] - token embeddings
 *   - selected_experts: [num_tokens, k] - expert assignments
 *   - num_experts: total number of experts
 * 
 * Output:
 *   - expert_inputs: [num_experts][max_tokens_per_expert, hidden_dim] - grouped tokens
 *   - expert_counts: [num_experts] - number of tokens assigned to each expert
 *   - token_to_expert_map: [num_tokens, k] - mapping for gather phase
 */
class TokenScatterKernel {
public:
    void operator()(sycl::queue& q,
                   const float* hidden_states,
                   const int* selected_experts,
                   float* expert_inputs,
                   int* expert_counts,
                   int* token_to_expert_map,
                   int num_tokens,
                   int hidden_dim,
                   int num_experts,
                   int k,
                   int max_tokens_per_expert) {
        
        // Allocate device memory
        auto states_dev = sycl::malloc_device<float>(num_tokens * hidden_dim, q);
        auto experts_dev = sycl::malloc_device<int>(num_tokens * k, q);
        auto counts_dev = sycl::malloc_device<int>(num_experts, q);
        auto inputs_dev = sycl::malloc_device<float>(num_experts * max_tokens_per_expert * hidden_dim, q);
        auto map_dev = sycl::malloc_device<int>(num_tokens * k, q);
        
        // Copy inputs
        q.memcpy(states_dev, hidden_states, num_tokens * hidden_dim * sizeof(float));
        q.memcpy(experts_dev, selected_experts, num_tokens * k * sizeof(int));
        q.memset(counts_dev, 0, num_experts * sizeof(int));
        q.wait();
        
        // Count tokens per expert (atomic)
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(num_tokens * k), [=](sycl::id<1> idx) {
                int expert_idx = experts_dev[idx[0]];
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> 
                    count(counts_dev[expert_idx]);
                count.fetch_add(1);
            });
        }).wait();
        
        // Scatter tokens (simplified - actual implementation needs more careful indexing)
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(num_tokens), [=](sycl::id<1> token_idx) {
                const int tid = token_idx[0];
                const float* token_state = states_dev + tid * hidden_dim;
                
                for (int i = 0; i < k; i++) {
                    int expert_idx = experts_dev[tid * k + i];
                    
                    // Get position in expert's input buffer (atomic increment)
                    // This is simplified - production code needs proper synchronization
                    int pos = tid % max_tokens_per_expert;  // Placeholder
                    
                    // Copy token to expert's input buffer
                    float* expert_input = inputs_dev + 
                        (expert_idx * max_tokens_per_expert + pos) * hidden_dim;
                    
                    for (int d = 0; d < hidden_dim; d++) {
                        expert_input[d] = token_state[d];
                    }
                    
                    // Record mapping for gather phase
                    map_dev[tid * k + i] = expert_idx * max_tokens_per_expert + pos;
                }
            });
        }).wait();
        
        // Copy results back
        q.memcpy(expert_inputs, inputs_dev, 
                num_experts * max_tokens_per_expert * hidden_dim * sizeof(float));
        q.memcpy(expert_counts, counts_dev, num_experts * sizeof(int));
        q.memcpy(token_to_expert_map, map_dev, num_tokens * k * sizeof(int));
        q.wait();
        
        // Free device memory
        sycl::free(states_dev, q);
        sycl::free(experts_dev, q);
        sycl::free(counts_dev, q);
        sycl::free(inputs_dev, q);
        sycl::free(map_dev, q);
    }
};

/**
 * Token gather kernel
 * Gathers expert outputs and applies routing weights
 * 
 * Input:
 *   - expert_outputs: [num_experts][max_tokens_per_expert, hidden_dim] - expert results
 *   - routing_weights: [num_tokens, k] - routing weights from top-k
 *   - token_to_expert_map: [num_tokens, k] - mapping from scatter phase
 * 
 * Output:
 *   - final_outputs: [num_tokens, hidden_dim] - weighted sum of expert outputs
 */
class TokenGatherKernel {
public:
    void operator()(sycl::queue& q,
                   const float* expert_outputs,
                   const float* routing_weights,
                   const int* token_to_expert_map,
                   float* final_outputs,
                   int num_tokens,
                   int hidden_dim,
                   int k) {
        
        // Allocate device memory
        auto outputs_dev = sycl::malloc_device<float>(num_tokens * hidden_dim, q);
        auto weights_dev = sycl::malloc_device<float>(num_tokens * k, q);
        auto map_dev = sycl::malloc_device<int>(num_tokens * k, q);
        
        // Copy inputs
        q.memcpy(weights_dev, routing_weights, num_tokens * k * sizeof(float));
        q.memcpy(map_dev, token_to_expert_map, num_tokens * k * sizeof(int));
        q.wait();
        
        // Gather and weight
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(num_tokens, hidden_dim), 
                          [=](sycl::id<2> idx) {
                const int tid = idx[0];
                const int dim = idx[1];
                
                float accumulated = 0.0f;
                
                // Accumulate weighted expert outputs
                for (int i = 0; i < k; i++) {
                    int expert_output_idx = map_dev[tid * k + i];
                    float weight = weights_dev[tid * k + i];
                    float expert_value = expert_outputs[expert_output_idx * hidden_dim + dim];
                    
                    accumulated += weight * expert_value;
                }
                
                outputs_dev[tid * hidden_dim + dim] = accumulated;
            });
        }).wait();
        
        // Copy results back
        q.memcpy(final_outputs, outputs_dev, num_tokens * hidden_dim * sizeof(float)).wait();
        
        // Free device memory
        sycl::free(outputs_dev, q);
        sycl::free(weights_dev, q);
        sycl::free(map_dev, q);
    }
};

/**
 * Batched expert FFN kernel
 * Processes multiple tokens through a single expert using batched GEMM
 * 
 * This is a simplified version - production should use vendor GEMM (oneDNN/MKL)
 * 
 * Input:
 *   - inputs: [batch_size, hidden_dim]
 *   - w1: [hidden_dim, ffn_dim] - up-projection
 *   - w2: [ffn_dim, hidden_dim] - down-projection
 * 
 * Output:
 *   - outputs: [batch_size, hidden_dim]
 */
class BatchedExpertFFNKernel {
public:
    void operator()(sycl::queue& q,
                   const float* inputs,
                   const float* w1,
                   const float* w2,
                   float* outputs,
                   int batch_size,
                   int hidden_dim,
                   int ffn_dim) {
        
        // Allocate intermediate buffer for activation
        auto intermediate = sycl::malloc_device<float>(batch_size * ffn_dim, q);
        
        // Up-projection: intermediate = inputs @ w1
        // This is a placeholder - use vendor GEMM (oneMKL) in production
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(batch_size, ffn_dim),
                          [=](sycl::id<2> idx) {
                const int b = idx[0];
                const int f = idx[1];
                
                float sum = 0.0f;
                for (int h = 0; h < hidden_dim; h++) {
                    sum += inputs[b * hidden_dim + h] * w1[h * ffn_dim + f];
                }
                
                // SiLU activation: x * sigmoid(x)
                float sigmoid = 1.0f / (1.0f + sycl::exp(-sum));
                intermediate[b * ffn_dim + f] = sum * sigmoid;
            });
        }).wait();
        
        // Down-projection: outputs = intermediate @ w2
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(batch_size, hidden_dim),
                          [=](sycl::id<2> idx) {
                const int b = idx[0];
                const int h = idx[1];
                
                float sum = 0.0f;
                for (int f = 0; f < ffn_dim; f++) {
                    sum += intermediate[b * ffn_dim + f] * w2[f * hidden_dim + h];
                }
                
                outputs[b * hidden_dim + h] = sum;
            });
        }).wait();
        
        sycl::free(intermediate, q);
    }
};

} // namespace moe_kernels

#endif // ENABLE_SYCL
