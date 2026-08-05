#pragma once

// Registers the Vulkan implementation of the MoE expert-cache bridge.
// Registration is opt-in through GGML_VK_MOE_CACHE=1.
void ggml_vk_moe_cache_register(void);
