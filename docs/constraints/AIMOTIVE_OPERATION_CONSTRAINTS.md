# AImotive Operation Support List

**Config:** Apache6 1600 MHz  
**Format:** ONNX

---

## Tensor Introducing Operations

| Row Id | Operation Name | Operation Constraints | Performance Considerations | Link |
|--------|----------------|----------------------|---------------------------|------|
| 1 | Constant | - | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Constant.html) |
| 2 | ConstantOfShape | - | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html) |

---

## Element-wise Operations

| Row Id | Operation Name | Operation Constraints | Performance Considerations | Link |
|--------|----------------|----------------------|---------------------------|------|
| 1 | Add | Two variable inputs or one constant input with shape (1,C) or (1,C,1,1), where C is either 1 or the channel size of the other input | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Add.html) |
| 2 | Div | Two variable inputs or one constant input with shape (1,C) or (1,C,1,1), where C is either 1 or the channel size of the other input | - | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Div.html) |
| 3 | HardSigmoid | - | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__HardSigmoid.html) |
| 4 | Mean | - | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Mean.html) |
| 5 | Mul | Two variable inputs or one constant input with shape (1,C) or (1,C,1,1), where C is either 1 or the channel size of the other input | - | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Mul.html) |
| 6 | Pow | Pow second argument must be constant | - | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Pow.html) |
| 7 | Softsign | - | - | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Softsign.html) |
| 8 | Sub | Two variable inputs or one constant input with shape (1,C) or (1,C,1,1), where C is either 1 or the channel size of the other input | - | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Sub.html) |
| 9 | Sum | Two variable inputs or one constant input with shape (1,C) or (1,C,1,1), where C is either 1 or the channel size of the other input | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Sum.html) |

---

## Sliding Window Operations

| Row Id | Operation Name | Operation Constraints | Performance Considerations | Link |
|--------|----------------|----------------------|---------------------------|------|
| 1 | Conv | Max window size: 17, max dilation: 5, stride: powers of 2 | Output channel count should be multiple of NNU_count | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Conv.html) |
| 2 | ConvTranspose | Max window size: 5, max dilation: 5, stride: powers of 2 | Output channel count should be multiple of NNU_count | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html) |

---

## Reduce Operations

| Row Id | Operation Name | Operation Constraints | Performance Considerations | Link |
|--------|----------------|----------------------|---------------------------|------|
| 1 | ArgMax | Axes must be [1], input channels must not exceed 256 | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__ArgMax.html) |
| 2 | GlobalAveragePool | - | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html) |
| 3 | GlobalMaxPool | - | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__GlobalMaxPool.html) |
| 4 | ReduceMax | Axes must only contain values of 1, 2 or 3 | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__ReduceMax.html) |
| 5 | ReduceMean | Axes must only contain values of 1, 2 or 3 | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__ReduceMean.html) |

---

## Tensor Shape Operations

| Row Id | Operation Name | Operation Constraints | Performance Considerations | Link |
|--------|----------------|----------------------|---------------------------|------|
| 1 | Concat | Axis must be 1, 2 or 3 | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Concat.html) |
| 2 | Flatten | Reshape: Not supported | W,H dimensions of output tensor should be multiply of 8 and leave padding on auto | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Flatten.html) |
| 3 | Pad | Either channel-wise padding (all zeros, except padding[1]) or all consumer operations are sliding window type (e.g. conv, max_pool, avg_pool) with no padding | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Pad.html) |
| 4 | Reshape | Reshape: Not supported | W,H dimensions of output tensor should be multiply of 8 and leave padding on auto | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Reshape.html) |
| 5 | Slice | Axes must be [1], [2] or [3], begin and end must be 1 element array which element divisible by 8, all stride values must be greater than 0 | W,H dimensions of output tensor should be multiply of 8 and leave padding on auto | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Slice.html) |
| 6 | Split | Axis must be 1, 2 or 3 | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Split.html) |
| 7 | Squeeze | Axes must be [2,3] | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Squeeze.html) |
| 8 | Unsqueeze | Axes must be [2,3] | W,H dimensions of output tensor should be multiply of 8 and leave padding on auto | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html) |

---

## Resizer Operations

| Row Id | Operation Name | Operation Constraints | Performance Considerations | Link |
|--------|----------------|----------------------|---------------------------|------|
| 1 | Upsample | If method = 'bilinear' or 'nearest', supported factor: powers of 2 | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Upsample.html) |

---

## Activation Operations

| Row Id | Operation Name | Link |
|--------|----------------|------|
| 1 | Abs | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Abs.html) |
| 2 | Acos | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Acos.html) |
| 3 | Acosh | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Acosh.html) |
| 4 | Asin | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Asin.html) |
| 5 | Asinh | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Asinh.html) |
| 6 | Atan | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Atan.html) |
| 7 | Atanh | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Atanh.html) |
| 8 | Ceil | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Ceil.html) |
| 9 | Clip | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Clip.html) |
| 10 | Cos | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Cos.html) |
| 11 | Cosh | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Cosh.html) |
| 12 | Elu | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Elu.html) |
| 13 | Exp | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Exp.html) |
| 14 | Floor | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Floor.html) |
| 15 | HardSigmoid | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__HardSigmoid.html) |
| 16 | LeakyRelu | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__LeakyRelu.html) |
| 17 | Log | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Log.html) |
| 18 | LpPool | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__LpPool.html) |
| 19 | Neg | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Neg.html) |
| 20 | PRelu | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__PRelu.html) |
| 21 | Reciprocal | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Reciprocal.html) |
| 22 | Relu | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Relu.html) |
| 23 | Selu | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Selu.html) |
| 24 | Sigmoid | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Sigmoid.html) |
| 25 | Sign | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Sign.html) |
| 26 | Sin | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Sin.html) |
| 27 | Sinh | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Sinh.html) |
| 28 | Softplus | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Softplus.html) |
| 29 | Softsign | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Softsign.html) |
| 30 | Sqrt | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Sqrt.html) |
| 31 | Tan | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Tan.html) |
| 32 | Tanh | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Tanh.html) |

---

## Pooling Operations

| Row Id | Operation Name | Operation Constraints | Performance Considerations | Link |
|--------|----------------|----------------------|---------------------------|------|
| 1 | AveragePool | Max window size: 17, max dilation: 5, supported stride: powers of 2 | W,H dimensions of output tensor should be multiply of 8 and leave padding on auto | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__AveragePool.html) |
| 2 | LpPool | If p == 1 or 2, max window size: 17, max dilation: 5, supported stride: powers of 2 | W,H dimensions of output tensor should be multiply of 8 and leave padding on auto | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__LpPool.html) |
| 3 | MaxPool | Max window size: 17, max dilation: 5, supported stride: powers of 2 | W,H dimensions of output tensor should be multiply of 8 and leave padding on auto | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__MaxPool.html) |

---

## Normalization Operations

| Row Id | Operation Name | Operation Constraints | Performance Considerations | Link |
|--------|----------------|----------------------|---------------------------|------|
| 1 | BatchNormalization | - | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) |
| 2 | LRN | - | W, H dimensions of the output tensor should be divisible by 8 | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__LRN.html) |

---

## Other Operations

| Row Id | Operation Name | Operation Constraints | Performance Considerations | Link |
|--------|----------------|----------------------|---------------------------|------|
| 1 | Cast | Output dtype must be 'scalar' (float32) | - | [ONNX Docs](https://onnx.ai/onnx/operators/onnx__Cast.html) |

---

## Key Constraints Summary

### Critical Hard Constraints
- **Slice**: Begin and end must be divisible by 8
- **Split**: Axis must be 1, 2 or 3
- **Concat**: Axis must be 1, 2 or 3
- **Conv**: Max window size 17, max dilation 5, stride must be powers of 2
- **ConvTranspose**: Max window size 5, max dilation 5, stride must be powers of 2
- **Pooling**: Max window size 17, max dilation 5, stride must be powers of 2
- **Upsample**: Factor must be powers of 2

### Performance Recommendations
- **W, H dimensions**: Should be divisible by 8 for optimal performance
- **Conv/ConvTranspose output channels**: Should be multiple of NNU_count
- **Reshape/Flatten**: Not supported - avoid if possible

---

**Copyright 2023 AImotive. All rights reserved.**
