#define CausalConv1d DcutCausalConv1d

// Include the CANN header FIRST so its IMPL_OP_OPTILING definition is processed
// and the include guard INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_H_ is set.
// Otherwise the header (pulled in transitively by the tiling file) would
// re-define IMPL_OP_OPTILING and clobber our override below.
#include "register/op_impl_registry.h"

// Override IMPL_OP_OPTILING to force macro expansion in #op_type (stringification)
// and ##op_type (token pasting). Per C standard, # and ## do NOT expand macro
// arguments, so without this override IMPL_OP_OPTILING(CausalConv1d) would
// register the op type as "CausalConv1d" instead of "DcutCausalConv1d", and the
// runtime would fail with "Do not find tiling func of DcutCausalConv1d!".
#undef IMPL_OP_OPTILING
#define DCUT_STRINGIFY(x) #x
#define DCUT_STRINGIFY_EXPAND(x) DCUT_STRINGIFY(x)
#define DCUT_CAT(a, b) DCUT_CAT_INNER(a, b)
#define DCUT_CAT_INNER(a, b) a##b
#define IMPL_OP_OPTILING(op_type) \
  gert::OpImplRegisterV2 VAR_UNUSED DCUT_CAT(op_impl_register_optiling_, op_type) = gert::OpImplRegisterV2(DCUT_STRINGIFY_EXPAND(op_type))

// The torch adapter requires query_start_loc, so this operator always knows
// the real request boundaries and must never let the host re-derive them from
// the shapes. Without this the packed varlen batch the D-Cut speculative path
// passes -- one request holding the whole verification window beside empty
// rows -- is read as one token per request whenever the token count equals the
// request count, which the FULL graph's fixed capture shapes make routine.
#define CAUSAL_CONV1D_QUERY_START_LOC_DEFINES_LAYOUT 1

#include "../../../../csrc/moe/causal_conv1d/op_host/causal_conv1d_tiling.cpp"
