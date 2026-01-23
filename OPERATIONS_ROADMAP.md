# LabNeura Operations Roadmap

Comprehensive list of operations to implement, based on NumPy and PyTorch APIs.

## Currently Supported Operations (v0.2.4)

### ✅ Basic Arithmetic
- `add` / `add_inplace` - Element-wise addition
- `sub` / `sub_inplace` - Element-wise subtraction  
- `mul` / `mul_inplace` - Element-wise multiplication

### ✅ Quantization Support
- FP32 (32-bit float)
- FP16 (16-bit float)
- INT16 (16-bit integer)
- INT8 (8-bit integer)

### ✅ Backend Detection
- AVX2 (x86_64)
- NEON (ARM64/Apple Silicon)
- SSE4.1 (x86_64 fallback)
- Generic (portable C++)

---

## 🎯 Priority 1: Core Arithmetic Operations (Next Release)

### Division
- **NumPy**: `np.divide()`, `/`, `//` (floor division)
- **PyTorch**: `torch.div()`, `torch.floor_divide()`
- **LabNeura**: `div()`, `div_inplace()`, `floor_div()`, `floor_div_inplace()`

### Power/Exponentiation
- **NumPy**: `np.power()`, `**`, `np.square()`, `np.sqrt()`
- **PyTorch**: `torch.pow()`, `torch.square()`, `torch.sqrt()`
- **LabNeura**: `pow()`, `pow_inplace()`, `square()`, `sqrt()`

### Absolute Value & Sign
- **NumPy**: `np.abs()`, `np.absolute()`, `np.sign()`
- **PyTorch**: `torch.abs()`, `torch.sign()`
- **LabNeura**: `abs()`, `abs_inplace()`, `sign()`

### Negation
- **NumPy**: `-x`, `np.negative()`
- **PyTorch**: `-x`, `torch.neg()`
- **LabNeura**: `neg()`, `neg_inplace()`

---

## 🎯 Priority 2: Reduction Operations

### Sum & Mean
- **NumPy**: `np.sum()`, `np.mean()`, `np.average()`
- **PyTorch**: `torch.sum()`, `torch.mean()`
- **LabNeura**: `sum()`, `mean()`, `sum_axis()`, `mean_axis()`

### Min & Max
- **NumPy**: `np.min()`, `np.max()`, `np.argmin()`, `np.argmax()`
- **PyTorch**: `torch.min()`, `torch.max()`, `torch.argmin()`, `torch.argmax()`
- **LabNeura**: `min()`, `max()`, `argmin()`, `argmax()`

### Product
- **NumPy**: `np.prod()`, `np.cumprod()`
- **PyTorch**: `torch.prod()`, `torch.cumprod()`
- **LabNeura**: `prod()`, `cumprod()`

### Cumulative Sum
- **NumPy**: `np.cumsum()`
- **PyTorch**: `torch.cumsum()`
- **LabNeura**: `cumsum()`

### Variance & Standard Deviation
- **NumPy**: `np.var()`, `np.std()`
- **PyTorch**: `torch.var()`, `torch.std()`
- **LabNeura**: `var()`, `std()`

---

## 🎯 Priority 3: Comparison & Logical Operations

### Element-wise Comparisons
- **NumPy**: `>`, `<`, `>=`, `<=`, `==`, `!=`
- **PyTorch**: `torch.gt()`, `torch.lt()`, `torch.ge()`, `torch.le()`, `torch.eq()`, `torch.ne()`
- **LabNeura**: `gt()`, `lt()`, `ge()`, `le()`, `eq()`, `ne()`

### Logical Operations
- **NumPy**: `np.logical_and()`, `np.logical_or()`, `np.logical_not()`, `np.logical_xor()`
- **PyTorch**: `torch.logical_and()`, `torch.logical_or()`, `torch.logical_not()`, `torch.logical_xor()`
- **LabNeura**: `logical_and()`, `logical_or()`, `logical_not()`, `logical_xor()`

### Min/Max Comparison
- **NumPy**: `np.minimum()`, `np.maximum()`
- **PyTorch**: `torch.minimum()`, `torch.maximum()`
- **LabNeura**: `minimum()`, `maximum()`

### Clamping
- **NumPy**: `np.clip()`
- **PyTorch**: `torch.clamp()`, `torch.clip()`
- **LabNeura**: `clamp()`, `clamp_inplace()`

---

## 🎯 Priority 4: Mathematical Functions

### Exponential & Logarithm
- **NumPy**: `np.exp()`, `np.log()`, `np.log10()`, `np.log2()`, `np.exp2()`
- **PyTorch**: `torch.exp()`, `torch.log()`, `torch.log10()`, `torch.log2()`
- **LabNeura**: `exp()`, `log()`, `log10()`, `log2()`, `exp2()`

### Trigonometric Functions
- **NumPy**: `np.sin()`, `np.cos()`, `np.tan()`, `np.arcsin()`, `np.arccos()`, `np.arctan()`
- **PyTorch**: `torch.sin()`, `torch.cos()`, `torch.tan()`, `torch.asin()`, `torch.acos()`, `torch.atan()`
- **LabNeura**: `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`

### Hyperbolic Functions
- **NumPy**: `np.sinh()`, `np.cosh()`, `np.tanh()`
- **PyTorch**: `torch.sinh()`, `torch.cosh()`, `torch.tanh()`
- **LabNeura**: `sinh()`, `cosh()`, `tanh()`

### Rounding
- **NumPy**: `np.round()`, `np.floor()`, `np.ceil()`, `np.trunc()`
- **PyTorch**: `torch.round()`, `torch.floor()`, `torch.ceil()`, `torch.trunc()`
- **LabNeura**: `round()`, `floor()`, `ceil()`, `trunc()`

---

## 🎯 Priority 5: Matrix Operations (Multi-Dimensional Support)

### Shape Manipulation
- **NumPy**: `np.reshape()`, `np.flatten()`, `np.ravel()`, `np.squeeze()`, `np.expand_dims()`
- **PyTorch**: `torch.reshape()`, `torch.flatten()`, `torch.squeeze()`, `torch.unsqueeze()`
- **LabNeura**: `reshape()`, `flatten()`, `squeeze()`, `unsqueeze()`

### Transpose & Permute
- **NumPy**: `np.transpose()`, `np.swapaxes()`
- **PyTorch**: `torch.transpose()`, `torch.permute()`, `torch.t()`
- **LabNeura**: `transpose()`, `permute()`, `t()`

### Matrix Multiplication
- **NumPy**: `np.dot()`, `np.matmul()`, `@`
- **PyTorch**: `torch.mm()`, `torch.matmul()`, `@`
- **LabNeura**: `matmul()`, `mm()`, `dot()`

### Broadcasting
- **NumPy**: Automatic broadcasting
- **PyTorch**: Automatic broadcasting
- **LabNeura**: Implement broadcasting rules for all operations

---

## 🎯 Priority 6: Neural Network Operations

### Activation Functions
- **PyTorch**: `torch.nn.functional.relu()`, `torch.nn.functional.sigmoid()`, `torch.nn.functional.softmax()`
- **LabNeura**: `relu()`, `sigmoid()`, `softmax()`, `gelu()`, `silu()`

### Convolution
- **PyTorch**: `torch.nn.functional.conv1d()`, `torch.nn.functional.conv2d()`
- **LabNeura**: `conv1d()`, `conv2d()` (with SIMD optimization)

### Pooling
- **PyTorch**: `torch.nn.functional.max_pool2d()`, `torch.nn.functional.avg_pool2d()`
- **LabNeura**: `max_pool2d()`, `avg_pool2d()`

### Normalization
- **PyTorch**: `torch.nn.functional.batch_norm()`, `torch.nn.functional.layer_norm()`
- **LabNeura**: `batch_norm()`, `layer_norm()`

---

## 🎯 Priority 7: Indexing & Selection

### Slicing
- **NumPy**: `x[start:end:step]`
- **PyTorch**: `x[start:end:step]`
- **LabNeura**: `slice()`, `index()`

### Masking
- **NumPy**: `x[mask]` (boolean indexing)
- **PyTorch**: `torch.masked_select()`
- **LabNeura**: `masked_select()`, `where()`

### Concatenation & Stacking
- **NumPy**: `np.concatenate()`, `np.stack()`, `np.hstack()`, `np.vstack()`
- **PyTorch**: `torch.cat()`, `torch.stack()`
- **LabNeura**: `cat()`, `stack()`, `hstack()`, `vstack()`

### Split
- **NumPy**: `np.split()`, `np.array_split()`
- **PyTorch**: `torch.split()`, `torch.chunk()`
- **LabNeura**: `split()`, `chunk()`

---

## 🎯 Priority 8: Random Number Generation

### Distributions
- **NumPy**: `np.random.rand()`, `np.random.randn()`, `np.random.uniform()`, `np.random.normal()`
- **PyTorch**: `torch.rand()`, `torch.randn()`, `torch.uniform_()`, `torch.normal_()`
- **LabNeura**: `rand()`, `randn()`, `uniform()`, `normal()`

### Random Sampling
- **NumPy**: `np.random.choice()`, `np.random.shuffle()`
- **PyTorch**: `torch.multinomial()`, `torch.randperm()`
- **LabNeura**: `choice()`, `shuffle()`, `randperm()`

---

## 🎯 Priority 9: Advanced Operations

### Sorting
- **NumPy**: `np.sort()`, `np.argsort()`
- **PyTorch**: `torch.sort()`, `torch.argsort()`
- **LabNeura**: `sort()`, `argsort()`

### Unique Values
- **NumPy**: `np.unique()`
- **PyTorch**: `torch.unique()`
- **LabNeura**: `unique()`

### Histogram
- **NumPy**: `np.histogram()`, `np.bincount()`
- **PyTorch**: `torch.histc()`, `torch.bincount()`
- **LabNeura**: `histogram()`, `bincount()`

### Quantile/Percentile
- **NumPy**: `np.quantile()`, `np.percentile()`, `np.median()`
- **PyTorch**: `torch.quantile()`, `torch.median()`
- **LabNeura**: `quantile()`, `percentile()`, `median()`

---

## 🎯 Priority 10: Linear Algebra Operations

### Norms
- **NumPy**: `np.linalg.norm()`
- **PyTorch**: `torch.linalg.norm()`, `torch.norm()`
- **LabNeura**: `norm()`, `vector_norm()`, `matrix_norm()`

### Matrix Decomposition
- **NumPy**: `np.linalg.svd()`, `np.linalg.eig()`, `np.linalg.qr()`
- **PyTorch**: `torch.linalg.svd()`, `torch.linalg.eig()`, `torch.linalg.qr()`
- **LabNeura**: `svd()`, `eig()`, `qr()` (future: use BLAS/LAPACK)

### Matrix Inverse & Solve
- **NumPy**: `np.linalg.inv()`, `np.linalg.solve()`
- **PyTorch**: `torch.linalg.inv()`, `torch.linalg.solve()`
- **LabNeura**: `inv()`, `solve()`

---

## Implementation Strategy

### Phase 1 (v0.3.0): Core Arithmetic Expansion
- Division, power, absolute value, negation
- All with SIMD optimization (AVX2/NEON/Generic)
- Python bindings + comprehensive tests

### Phase 2 (v0.4.0): Reductions & Comparisons
- Sum, mean, min, max, argmin, argmax
- Element-wise comparisons
- Logical operations

### Phase 3 (v0.5.0): Mathematical Functions
- Exponential, logarithm, trigonometric
- Rounding functions
- Fast approximations using SIMD

### Phase 4 (v0.6.0): Multi-Dimensional Support
- Shape manipulation
- Matrix multiplication (optimized with SIMD)
- Broadcasting support

### Phase 5 (v0.7.0): Neural Network Primitives
- Activation functions (ReLU, Sigmoid, Softmax, GELU)
- Basic convolution operations
- Pooling operations

### Phase 6 (v0.8.0+): Advanced Features
- Indexing, slicing, masking
- Random number generation
- Linear algebra operations

---

## SIMD Optimization Targets

All operations should have optimized implementations for:
- **AVX2**: 8 FP32 / 16 INT16 / 32 INT8 per instruction
- **NEON**: 4 FP32 / 8 INT16 / 16 INT8 per instruction
- **Generic**: Portable scalar fallback

### Critical Fast Paths
1. **Activation Functions**: ReLU, Sigmoid - used in every neural network layer
2. **Matrix Multiplication**: Core operation for deep learning
3. **Reductions**: Sum/Mean - frequent in normalization layers
4. **Comparisons**: Used in ReLU, clamp, etc.

---

## Testing Requirements

Each operation must have:
1. ✅ Unit tests for all quantization modes (FP32, FP16, INT16, INT8)
2. ✅ Tests for all backends (AVX2, NEON, Generic)
3. ✅ Edge cases: empty tensors, size=1, large tensors
4. ✅ Numerical accuracy validation vs NumPy/PyTorch
5. ✅ Performance benchmarks

---

## API Design Principles

1. **NumPy-style naming** where possible for familiarity
2. **In-place variants** (`_inplace` suffix) for memory efficiency
3. **Automatic backend selection** based on hardware
4. **Python/C++ API parity** - same operations available in both
5. **Shape broadcasting** following NumPy rules
6. **Type promotion** rules consistent with NumPy/PyTorch

---

## Documentation Updates Needed

- [ ] API reference for each operation
- [ ] Performance comparison charts (LabNeura vs NumPy)
- [ ] SIMD implementation details per operation
- [ ] Migration guide from NumPy/PyTorch
- [ ] Quantization accuracy analysis per operation

---

**Total Operations to Implement**: ~150+ operations across all priorities

**Estimated Timeline**: 
- Priority 1-3: 2-3 months
- Priority 4-6: 3-4 months  
- Priority 7-10: 4-6 months
- Total: ~12 months for comprehensive coverage
