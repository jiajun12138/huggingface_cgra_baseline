import torch, math

frac_bits = {8:3, 16: 9, 32: 9}

def get_minq_maxq(bits: int, sym: bool):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = torch.tensor(-maxq -1)
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = torch.tensor(0)

    return minq, maxq

def asym_quantize(x: torch.Tensor, bits: int):
    minq, maxq = get_minq_maxq(bits=bits, sym=False)
    xmax = torch.amax(x, dim=-1, keepdim=True)
    xmin = torch.zeros_like(xmax)
    # print("xmax, xmin", xmax, xmin, x.max(), x.min(), maxq)
    # print("sub", xmax - xmin, "max", (xmax - xmin).max())
    # print("clamp", ((xmax - xmin)*0.9).clamp(min=1e-5))
    # print("clamp result", ((xmax - xmin)*0.9).max())
    scale = (xmax - xmin)*0.9
    if scale.max() == 0:
        scale = torch.tensor(1.0)
    # print("scale in this, ", scale, scale.max())
    zero = torch.zeros_like(xmax)
    q = torch.clamp(torch.round((x + zero) / scale), -xmax, xmax)

    return q, scale, zero

def asym_dequantize(q, scale, zero):
    return q * scale - zero

def frac_mult(x, y, bw):
    #print(x)
    # print(x,y)
    scale = frac_bits[bw]
    tmp_x=(x*(2**(scale-1))).to(torch.int64)
    # print('x: ', x)
    # print(y)
    tmp_y=(y*(2**(scale-1))).to(torch.int64)
    # print('y: ', y)
    ans = (tmp_x * tmp_y).to(torch.int64)
    if(ans >= 2 ** ((2 * bw) - 2)).any():
        print('multiplication overflow', 2 ** ((2 * bw) - 2),x, y, ans)
    ans[ans >= 2 ** (2 * bw - 2)] = (2 ** (2 * bw - 2)) - 2
    result = (ans/(2**(scale-1))).to(torch.int64)
    return result/(2**(scale-1))

def frac_exp2(x, bw, term):
    # q, scale, zero = asym_quantize(x, bw)
    # result = torch.zeros_like(x)
    # factorial = 1
    ln2 = torch.log(torch.tensor(2))
    scale1 = ln2
    q1 = 1.5 / scale1
    scale2 = scale1 ** 2 / 6
    q2 = 0.625 / scale2
    scale3 = scale2 * ln2
    q3 = 1.0 / scale3
    if term == 3:
        tmp1 = frac_add(x, q1, bw)
        tmp2 = frac_mult(tmp1, tmp1, bw)
        tmp3 = frac_add(tmp2, q2, bw)
        tmp4 = frac_mult(tmp3, x, bw)
        result = frac_add(tmp4, q3, bw)
    else:
        assert False

    return result, scale3

count = {"1":0}

def custom_int_exp(x, bw, term):
    # print(x)
    input = x * torch.tensor(1.442695)
    # count["1"] += 1
    # if count["1"] <= 5:
    #     print("input:", input, input.max(), input.min())

    # _, scale, zero = asym_quantize(input, bw)
    # if scale.max() in [float('inf'), float('-inf')]:
        # print('scale overflow', scale, scale.max(), x.max(), x.min())
    int_part = torch.floor(input)
    frac_part = input - int_part
    #print(frac_part)
    # print(int_part)
    max_int_scale = 2 ** int(input.max() * 0.9)
    # print(input.max(), max_int_scale)
    q, scale = frac_exp2(frac_part, bw, term)
    q = q * torch.pow(2, int_part) / max_int_scale
    return q, scale * max_int_scale
    
def frac_add(x, y, bw):
    #print(x)
    scale = frac_bits[bw]
    tmp_x=(x*(2**(scale-1))).to(torch.int64)
    #print('x: ', x)
    #print(y)
    tmp_y=(y*(2**(scale-1))).to(torch.int64)
    #print('y: ', y)
    ans = tmp_x + tmp_y
    # if(ans >=2 **(bw-1)).any():
    #   print('addition overflow')
    ans[ans >= 2 ** (bw - 1)] = (2 ** (bw - 1)) - 1
    result = ans.to(torch.int64)
    return result/(2**(scale-1))

def frac_div(x, y, bw):
    #print(x)
    scale = frac_bits[bw]
    tmp_x=(x*(2**(scale-1))).to(torch.int64)
    #print('x: ', x)
    #print(y)
    tmp_y=(y*(2**(scale-1))).to(torch.int64)
    # print(x, y, x/y)
    #print('y: ', y)
    return tmp_x / tmp_y

import math

def custom_int_tanh(x, bw, term):
    indices1, indices2 = x > 10, x < -10
    # print(x)
    x[indices1] = 0
    x[indices2] = 0
    exp_2x, scale = custom_int_exp(x * (-2.0), bw, term)
    q = 1.0 / scale
    tanh_x = frac_add(q, -exp_2x, bw) / frac_add(q, exp_2x, bw)
    tanh_x[indices1] = 1.0
    tanh_x[indices2] = -1.0
    return tanh_x

def custom_int_gelu(x, bw, term):

    # find the position in x that x[i] == inf / -inf
    # pos = torch.argwhere(torch.isinf(x))
    # count["1"] += 1
    # if count["1"] <= 5:
    #     print("x", x.max(), x.min(), x)
    # q, scale, zero = asym_quantize(x, bw)
    scale = x.abs().max() * 0.9
    q = x / scale
    
    scale1 = scale ** 2 * 0.044715
    q1 = 1.0 / scale1
    scale2 = scale1 * scale * (math.sqrt(2 / math.pi) ) 
    # print(1)
    x_2 = frac_mult(q, q, bw)
    # print(2)
    x_2_tmp = frac_add(x_2, q1, bw)
    # print(3)
    x_3 = frac_mult(q, x_2_tmp, bw)
    # print(4)

    # print(x_3 * scale1)
    tanh = custom_int_tanh(x_3 * scale2, bw, term)
    tanh_plus1 = frac_add(torch.tensor(1.0), tanh, bw)

    return frac_mult(q, tanh_plus1, bw) * scale * 0.5

def custom_int_softmax(x, bw, term):
    # print("softmax input", (torch.abs(x) >= 20000).any(), torch.isnan(x).any())
    new_x = x.to(torch.float64)
    # x_clamp = torch.clamp(new_x, min = - 20)
    x_max = torch.max(new_x, -1, keepdim=True)[0]
    x_norm = new_x - x_max
    # print("norm input", x_norm, torch.isnan(x_norm).any())
    print("softmax input", (torch.abs(x_norm) >= 20000).any(), torch.isnan(x_norm).any())
    x_exp, s = custom_int_exp(x_norm, bw, term)
    if torch.isnan(x_exp).any():
        print('x_exp overflow', x_exp.dtype)
    int_s = 2 ** frac_bits[bw]
    x_exp = (x_exp * int_s).to(torch.int64)
    x_sum = x_exp.sum(dim=-1, keepdim=True)
    if torch.isnan(x_sum).any():
        print('x_sum overflow', x_sum.dtype)
    # print("sum should be", x_exp / x_sum)

    return x_exp.to(torch.float64) / x_sum.to(torch.float64)
    # return frac_div(x_exp, x_sum, bw)
    # return x_exp / x_sum

def custom_int_layernorm(x, w, b, bw):
    if torch.isnan(x).any():
        print('before ln x overflow', x.dtype)
    eps = 1e-5
    # x_sum_x = torch.tensor(0)
    # x_sum_x2 = torch.tensor(0)
    # scale = x.max() * 0.9
    scale = x.max() * 0.9
    x_1 = x / scale
    # count["1"] += 1
    # if count["1"] <= 8:
    # print("statistics:", x.max() * 0.9)

    int_s = 2 ** frac_bits[bw]
    x_1 = (x_1 * int_s).to(torch.int64)

    N = x_1.shape[-1]
    # print(N)
    x_sum_x = x_1.sum(dim=-1, keepdim=True) / N
    x_sum_x2 = (x_1 ** 2).sum(dim=-1, keepdim=True) / N 
    # for x_i in x_1:
    #     x_sum_x = frac_add(x_sum_x, x_i, bw)
    #     x_sum_x2 = frac_add(frac_mult(x_i, x_i, bw), x_sum_x2, bw)
    # x_sum_x /= N
    # x_sum_x2 /= N
    if w is None:
        weight = 1.0
    else:
        weight = w
    if b is None:
        bias = 0.0
    else:
        bias = b
    invsqrt = 1.0 / (x_sum_x2 - (x_sum_x ** 2) + eps).sqrt()
    # print("statistics:")
    # print(x_1.max(), x_1.max(dim=-1), x_1.min(), x_1.min(dim=-1))
    # print(invsqrt.max(), invsqrt.min(), torch.isnan(invsqrt).any(), torch.isinf(invsqrt).any())
    # print(x_sum_x2.max(), x_sum_x2.min(), torch.isnan(x_sum_x2).any(), torch.isinf(x_sum_x2).any())
    # print(x_sum_x.max(), x_sum_x.min(), torch.isnan(x_sum_x).any(), torch.isinf(x_sum_x).any())
    # # prrint(invsqrt, 1.0 / (x_1.var()))
    # print("weight:")
    # print(w.max(), w.min(), torch.isnan(w).any(), torch.isinf(w).any())
    # print("bias:")
    # print(b.max(), b.min(), torch.isnan(b).any(), torch.isinf(b).any())
    # print("shape", w.shape, x_1.shape, x_sum_x.shape, b.shape, invsqrt.shape)
    ans = w * (x_1 - x_sum_x) * invsqrt + b
    if torch.isnan(ans.to(x.dtype)).any():
        print('ln overflow', ans.dtype)
    if (torch.abs(ans) >= 30000).any():
        print('ln overflow111', ans.dtype, ans.max(dim=-1), ans.max(), ans.min(), w, b)
    return ans.to(x.dtype)
