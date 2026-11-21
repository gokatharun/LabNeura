#include "labneura/tensor/tensor.h"

#include <vector>

int main() {
    labneura::Tensor left(std::vector<float>{1.0F, 2.0F});
    return left.size() == 2 && left.numel() == 2 ? 0 : 1;
}
