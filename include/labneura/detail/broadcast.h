#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace labneura::detail {

struct BroadcastPlan {
    std::vector<std::size_t> shape;
    std::vector<std::size_t> lhs_strides;
    std::vector<std::size_t> rhs_strides;
};

inline std::size_t checked_broadcast_numel(
    const std::vector<std::size_t>& shape) {
    if (std::find(shape.begin(), shape.end(), std::size_t{0}) != shape.end()) {
        return 0;
    }

    std::size_t numel = 1;
    for (const std::size_t dimension : shape) {
        if (numel > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error(
                "Broadcast output element count overflows size_t");
        }
        numel *= dimension;
    }
    return numel;
}

inline BroadcastPlan make_broadcast_plan(
    const std::vector<std::size_t>& lhs_shape,
    const std::vector<std::size_t>& lhs_strides,
    const std::vector<std::size_t>& rhs_shape,
    const std::vector<std::size_t>& rhs_strides) {
    if (lhs_shape.size() != lhs_strides.size()) {
        throw std::invalid_argument(
            "Left broadcast shape and stride ranks must match");
    }
    if (rhs_shape.size() != rhs_strides.size()) {
        throw std::invalid_argument(
            "Right broadcast shape and stride ranks must match");
    }

    const std::size_t output_rank =
        std::max(lhs_shape.size(), rhs_shape.size());
    BroadcastPlan plan{
        std::vector<std::size_t>(output_rank),
        std::vector<std::size_t>(output_rank, 0),
        std::vector<std::size_t>(output_rank, 0)};

    for (std::size_t output_axis = output_rank; output_axis > 0;
         --output_axis) {
        const std::size_t axis = output_axis - 1;
        const bool lhs_present = axis >= output_rank - lhs_shape.size();
        const bool rhs_present = axis >= output_rank - rhs_shape.size();
        const std::size_t lhs_axis =
            lhs_present ? axis - (output_rank - lhs_shape.size()) : 0;
        const std::size_t rhs_axis =
            rhs_present ? axis - (output_rank - rhs_shape.size()) : 0;
        const std::size_t lhs_dimension =
            lhs_present ? lhs_shape[lhs_axis] : 1;
        const std::size_t rhs_dimension =
            rhs_present ? rhs_shape[rhs_axis] : 1;

        std::size_t output_dimension;
        if (lhs_dimension == rhs_dimension) {
            output_dimension = lhs_dimension;
        } else if (lhs_dimension == 1) {
            output_dimension = rhs_dimension;
        } else if (rhs_dimension == 1) {
            output_dimension = lhs_dimension;
        } else {
            throw std::invalid_argument(
                "Cannot broadcast dimensions " +
                std::to_string(lhs_dimension) + " and " +
                std::to_string(rhs_dimension) + " at aligned axis " +
                std::to_string(axis));
        }

        plan.shape[axis] = output_dimension;
        if (lhs_present && lhs_dimension == output_dimension) {
            plan.lhs_strides[axis] = lhs_strides[lhs_axis];
        }
        if (rhs_present && rhs_dimension == output_dimension) {
            plan.rhs_strides[axis] = rhs_strides[rhs_axis];
        }
    }

    static_cast<void>(checked_broadcast_numel(plan.shape));
    return plan;
}

}  // namespace labneura::detail
