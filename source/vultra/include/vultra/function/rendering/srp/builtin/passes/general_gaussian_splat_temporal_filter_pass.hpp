#pragma once

#include "vultra/core/base/base.hpp"
#include "vultra/core/rhi/render_pass.hpp"
#include "vultra/core/rhi/structs/extent2d.hpp"
#include "vultra/core/rhi/structs/pixel_format.hpp"
#include "vultra/function/framegraph/framegraph_context.hpp"

#include <array>
#include <fg/Fwd.hpp>

namespace vultra
{
    namespace rhi
    {
        class Texture;
    }

    class GeneralGaussianSplatTemporalFilterPass final
        : public rhi::RenderPass<GeneralGaussianSplatTemporalFilterPass>
    {
    public:
        GeneralGaussianSplatTemporalFilterPass();

        FrameGraphResource filter(FrameGraphBuildContext& ctx, FrameGraphResource source);

        rhi::GraphicsPipeline createPipeline(rhi::PixelFormat colorFormat, bool useMultiview) const;

    private:
        void ensureTemporalHistory(rhi::RenderDevice& rd, rhi::Extent2D extent);
        void resetTemporalHistory();

    private:
        std::array<Ref<rhi::Texture>, 2> m_TemporalHistory {};
        rhi::Extent2D                    m_TemporalHistoryExtent {};
        uint32_t                         m_TemporalHistoryReadIndex {0u};
        bool                             m_TemporalHistoryValid {false};
    };
} // namespace vultra
