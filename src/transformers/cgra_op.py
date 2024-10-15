import torch, math

frac_bits = {8:3, 16: 10, 32: 9}

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
    if bw != 64:
        scale = frac_bits[bw]
    else:
        return x * y
    # tmp_x=(x*(2**(scale-1))).to(torch.int64)
    # print('x: ', x)
    # print(y)
    # tmp_y=(y*(2**(scale-1))).to(torch.int64)
    # print('y: ', y)
    # ans = (tmp_x * tmp_y).to(torch.int64)
    # if(ans >= 2 ** ((2 * bw) - 2)).any():
    #     print('multiplication overflow', 2 ** ((2 * bw) - 2),x, y, ans)
    # ans[ans >= 2 ** (2 * bw - 2)] = (2 ** (2 * bw - 2)) - 2
    # result = (ans/(2**(scale-1))).to(torch.int64)
    return ((x*(2**(scale-1))).to(torch.int64) * (y*(2**(scale-1))).to(torch.int64))/(2**(2*scale-2))

def frac_exp2(x, bw, term):
    # q, scale, zero = asym_quantize(x, bw)
    # result = torch.zeros_like(x)
    # factorial = 1
    ln2 = torch.log(torch.tensor(2))
    if bw != 64:
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
    else:
        if (torch.isnan(x)).any():
            print(x.dtype)
            assert False
        result = torch.zeros_like(x)
        power = torch.ones_like(x)
        factorial = 1
        for n in range(term):
            result += power / factorial
            power *= x
            power *= ln2
            factorial *= (n + 1)
        scale3 = 1

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
    # max_int_scale = 2 ** torch.floor(input.max(dim=-1, keepdim=True) * 0.9)
    # max_int_scale[max_int_scale > 2 ** 6] = 2 ** 6
    # count["1"] += 1
    # if count["1"] <= 5:
    #     print(input.max(), max_int_scale)
    q, scale = frac_exp2(frac_part, bw, term)
    if bw != 64:
        max_int_scale = 2 ** int(input.max() * 0.8)
        if max_int_scale > 2 ** 6:
            max_int_scale = 2 ** 6
        # max_int_scale = 2 ** torch.floor(torch.amax(input, dim=-1, keepdim=True) * 0.9)
        # max_int_scale[max_int_scale > 2 ** 6] = 2 ** 6
        q = q * torch.pow(2, int_part) / max_int_scale
        return q, scale * max_int_scale
    else:
        if (torch.isnan(q)).any():
            print(x.dtype)
            assert False
        return q * torch.pow(2, int_part), 1
    
def frac_add(x, y, bw):
    #print(x)
    if bw != 64:
        scale = frac_bits[bw]
    else:
        return x + y
    # tmp_x=(x*(2**(scale-1))).to(torch.int64)
    #print('x: ', x)
    #print(y)
    # tmp_y=(y*(2**(scale-1))).to(torch.int64)
    #print('y: ', y)
    # ans = tmp_x + tmp_y
    # if(ans >=2 **(bw-1)).any():
    #   print('addition overflow')
    # ans[ans >= 2 ** (bw - 1)] = (2 ** (bw - 1)) - 1
    # result = ans.to(torch.int64)
    return ((x*(2**(scale-1))).to(torch.int64) + (y*(2**(scale-1))).to(torch.int64))/(2**(scale-1))

def frac_div(x, y, bw):
    #print(x)
    if bw != 64:
        scale = frac_bits[bw]
    else:
        return x / y
    # tmp_x=(x*(2**(scale-1))).to(torch.int64)
    #print('x: ', x)
    #print(y)
    # tmp_y=(y*(2**(scale-1))).to(torch.int64)
    # print(x, y, x/y)
    #print('y: ', y)
    return (x*(2**(scale-1))).to(torch.int64) / (y*(2**(scale-1))).to(torch.int64)

import math

def custom_int_tanh(x, bw, term):
    # print(x.max(), x.min())
    # x[indices1] = 0
    # x[indices2] = 0
    exp_2x, scale = custom_int_exp(x * (-2.0), bw, term)
    q = 1.0 / scale
    tanh_x = frac_add(q, -exp_2x, bw) / frac_add(q, exp_2x, bw)
    # tanh_x[indices1] = 1.0
    # tanh_x[indices2] = -1.0
    return tanh_x

def custom_int_gelu(x, bw, term):

    # find the position in x that x[i] == inf / -inf
    # pos = torch.argwhere(torch.isinf(x))
    # count["1"] += 1
    # if count["1"] <= 5:
    #     print("x", x.max(), x.min(), x)
    # q, scale, zero = asym_quantize(x, bw)
    if torch.isnan(x).any() or (torch.abs(x) >= 30000).any() :
        print('before gelu overflow', x.dtype, x.max(dim=-1), x.max(), x.min())
    save_x = torch.clone(x)
    indices1 = math.sqrt(2.0 / math.pi) * x * (1.0 + 0.044715 * x ** 2) > 5
    indices2 = math.sqrt(2.0 / math.pi) * x * (1.0 + 0.044715 * x ** 2) < -5
    x[indices1] = 0
    x[indices2] = 0
    scale = x.abs().max()
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

    ans = frac_mult(q, tanh_plus1, bw) * scale * 0.5
    ans[indices1] = save_x[indices1].to(ans.dtype)
    ans[indices2] = 0

    if torch.isnan(ans.to(x.dtype)).any():
        print('gelu overflow', ans.dtype)
    if (torch.abs(ans) >= 30000).any():
        print('gelu overflow111', ans.dtype, ans.max(dim=-1), ans.max(), ans.min())
    return ans.to(x.dtype)

def custom_int_softmax(x, bw, term):
    # print("softmax input", (torch.abs(x) >= 20000).any(), torch.isnan(x).any())
    # return torch.nn.functional.softmax(x, dim=-1)
    return torch.nn.functional.softmax(x, dim=-1)
    new_x = x.to(torch.float64)
    # x_clamp = torch.clamp(new_x, min = - 20)
    x_max = torch.max(new_x, -1, keepdim=True)[0]
    x_norm = new_x - x_max
    # print("norm input", x_norm, torch.isnan(x_norm).any())
    # print("softmax input", (torch.abs(x_norm) >= 20000).any(), torch.isnan(x_norm).any())
    indices = x_norm < -10000
    x_norm[indices] = 0
    x_exp, s = custom_int_exp(x_norm, bw, term)
    x_exp[indices] = 0
    if torch.isnan(x_exp).any():
        print('x_exp overflow', x_exp.dtype)
    if bw != 64:
        int_s = 2 ** frac_bits[bw]
        x_exp = (x_exp * int_s).to(torch.int64)
        x_sum = x_exp.sum(dim=-1, keepdim=True)
    else:
        x_exp[indices] = 0.0
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
    scale = torch.amax(x, dim=-1, keepdim=True)
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

def custom_int_rmsnorm(x, w, eps, bw):
    # if count["1"] <= 5:
    #     print(x.max(), x.min(), w.max(), w.min())
    if torch.isnan(x).any():
        print('before ln x overflow', x.dtype)
    # x_sum_x = torch.tensor(0)
    # x_sum_x2 = torch.tensor(0)
    # scale = x.max() * 0.9
    # scale = torch.amax(x.abs(), dim=-1, keepdim=True) * 0.9
    # x_1 = x / torch.amax(x.abs(), dim=-1, keepdim=True) * 0.9
    # count["1"] += 1
    # if count["1"] <= 8:
    # print("statistics:", x.max() * 0.9)
    if bw != 64:
        int_s = 2 ** frac_bits[bw]
        x_1 = ((x / torch.amax(x.abs(), dim=-1, keepdim=True) * 0.9) * int_s).to(torch.int64)
    else:
        x_1 = x.to(torch.float64)
        variance = x_1.pow(2).mean(-1, keepdim=True)
        return (w * x_1 * torch.rsqrt(variance + eps)).to(x)
        # x_1 = x / torch.amax(x.abs(), dim=-1, keepdim=True) * 0.9

    N = x.shape[-1]
    # x_sum_x2 = (x_1 ** 2).sum(dim=-1, keepdim=True) / N 
    
    invsqrt = 1.0 / (((x_1 ** 2).sum(dim=-1, keepdim=True) / N ) + eps).sqrt()
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
    # if count["1"] <= 5:
    #     print("shape", w.shape, x_1.shape,  invsqrt.shape)
    #     print("invsqrt", invsqrt.max(), invsqrt.min())
        # print("x_sum_x2", x_sum_x2.max(), x_sum_x2.min())
    

    ans = w * (x_1) * invsqrt
    # if torch.isnan(ans.to(x.dtype)).any():
    #     print('ln overflow', invsqrt.max(), invsqrt.min(), x_sum_x2.max(), x_sum_x2.min())
    # if (torch.abs(ans) >= 30000).any():
    #     print('ln overflow111', ans.dtype, ans.max(dim=-1), ans.max(), ans.min(), w, x_1.min(), invsqrt.max(), invsqrt.min())
    return ans.to(x.dtype)

def frac_log2(ex, x, bw, term):
    # q, scale, zero = asym_quantize(x, bw)
    # result = torch.zeros_like(x)
    # factorial = 1
    ln2 = torch.log(torch.tensor(2))
    scale1 = torch.tensor(1.0)
    q1 = -0.75 / scale1
    scale2 = torch.tensor(1.0 / 3)
    q2 = 13.0 / 16 / scale2
    scale3 = scale2 / ln2
    q3 = ex / scale3
    if term == 3:
        tmp1 = frac_add(x, q1, bw)
        tmp2 = frac_mult(tmp1, tmp1, bw)
        tmp3 = frac_add(tmp2, q2, bw)
        tmp4 = frac_mult(tmp3, x, bw)
        result = frac_add(tmp4, q3, bw)
    else:
        assert False

    return result, scale3

def custom_int_log(x, bw, term):
    # print(x, x.dtype)
    e_x = torch.floor(torch.log(x) / torch.log(torch.tensor(2.0)))

    m_x = (x / torch.pow(2, e_x)) - 1
    # print(e_x, m_x)
    # print(2 ** e_x * (1+m_x))

    log2_e_x, scale = frac_log2(e_x, m_x, bw, term)
    # print(log2_e_x, scale)

    return log2_e_x, scale

def gemmlowp_silu(x, bw, term):

    # 1 + exp(-x)
    lut = [0.25, 0.5, 1.0, 2, 4, 8, 16]
    exp = torch.tensor([1672461947, 1302514674, 790015084, 290630308, 39332535, 720401, 242]) /  2 ** 31
    exp_0125 = torch.exp(torch.tensor(-0.125, dtype=torch.float16)) 
    ans = torch.zeros_like(x)
    indices1 = x >= 31.75

    # for k, x_i in enumerate(x):
    #     tmp = 1.0
    #     if x_i >= 31.75:
    #         # print("shabi")
    #         continue
    #     now = 0.0
    tmp = torch.zeros_like(x) + 1.0
    now = torch.zeros_like(x) 
    for i, entry in enumerate(lut):
        indices = now + entry <= x
        tmp[indices] *= exp[i]
        now[indices] += entry
    t = 0.125 - (x - now)
    indices = now != x
    tmp[indices] *= (exp_0125 * (1 + t + torch.pow(t, 2) / 2.0))[indices]
    ans = 1 + tmp
    ans[indices1] = 1.0
    
    return x / ans

def custom_int_silu(x, bw, term):
    return gemmlowp_silu(x, bw, term)
    # return torch.nn.functional.silu(x)
    # x * sigmoid(x)
    fp_x = x.to(torch.float64)
    o_scale = x.max() * 0.9

    indices1 = fp_x >= 6.0
    indices2 = fp_x <= -6.0
    fp_x[indices2] = 0.0
    exp_x, scale = custom_int_exp(-fp_x, bw, term)
    if bw != 64:
    # print("exp", exp_x * scale, torch.exp(-x))
        exp_x[indices1] = 0.0
        if exp_x[exp_x < 0.0].any():
            print('exp', exp_x.max(), exp_x.min(), exp_x.abs().min())

        # if scale.abs() > 2 ** 7:
        #     exp_x = exp_x * (scale / 2 ** 7)
        #     scale = torch.tensor(2 ** 7)
        if (scale > 2 ** 7).any():
            print("max_scale", scale.max())
        # indices = scale > 2 ** 7
        # exp_x[indices] = exp_x[indices] * (scale[indices] / 2 ** 7)
        # scale[indices] = 2 ** 7

        exp_plus1 = exp_x * scale + 1.0
        fp_x = (fp_x / exp_plus1).to(x.dtype)
        # exp_plus1 = frac_add(exp_x, torch.tensor(1.0) / scale, bw)

        # if exp_plus1[exp_plus1 <= 1.0 / scale].any():
        #     print('exp_plus', exp_plus1.max(), exp_plus1.min(), exp_plus1.abs().min())
        
        # exp_plus1[exp_plus1 <= 1.0 / scale] = 1.0 / scale

        # ans = (frac_div(fp_x / o_scale, exp_plus1, bw) * o_scale / scale).to(x.dtype)

        fp_x[indices2] = 0.0

    # if torch.isnan(ans).any() or (torch.abs(ans) >= 30000).any() or torch.isinf(ans).any():
    #     print('silu overflow', o_scale, scale, x.max(), x.abs().min(), x.min(), exp_plus1.max(), exp_plus1.min())
    #     print('ans', ans.max(), ans.min(), ans.abs().min(), torch.isnan(ans).any())
    
    else:
        fp_x /= (exp_x + 1.0)

    return fp_x.to(x)

