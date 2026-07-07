#pragma once

#include "vultra/core/base/base.hpp"
#include "vultra/core/rhi/render_pass.hpp"
#include "vultra/core/rhi/structs/extent2d.hpp"
#include "vultra/core/rhi/structs/pixel_format.hpp"
#include "vultra/function/framegraph/framegraph_context.hpp"

#include <fg/Fwd.hpp>

#include <array>

namespace vultra
{
    namespace rhi
    {
        class Texture;
    }

    class FinalCompositionPass final : public rhi::RenderPass<FinalCompositionPass>
    {
        friend class BasePass;

    public:
        FinalCompositionPass();
        FrameGraphResource compose(FrameGraphBuildContext& ctx, FrameGraphResource target);

    private:
        rhi::GraphicsPipeline createPipeline(rhi::PixelFormat colorFormat, bool useMultiview, bool temporalFilter) const;
        void ensureTemporalHistory(rhi::RenderDevice& rd, rhi::Extent2D extent);
        void resetTemporalHistory();

        std::array<Ref<rhi::Texture>, 2> m_TemporalHistory {};
        rhi::Extent2D                    m_TemporalHistoryExtent {};
        uint32_t                         m_TemporalHistoryReadIndex {0u};
        bool                             m_TemporalHistoryValid {false};
    };
} // namespace vultra
