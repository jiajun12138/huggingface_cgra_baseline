import torch, math

frac_bits = {8:3, 16: 7, 32: 8}

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
    print("clamp result", ((xmax - xmin)*0.9).max())
    scale = (xmax - xmin)*0.9
    print("scale in this, ", scale, scale.max())
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
    if(ans >= 2 **(2 * bw-1)).any():
        print('multiplication overflow')
    ans[ans >= 2 ** (2*bw - 1)] = (2 ** (2*bw - 1)) - 1
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

def custom_int_exp(x, bw, term):
    #print(fp_x)
    input = x*torch.tensor(1.442695)

    # _, scale, zero = asym_quantize(input, bw)
    # if scale.max() in [float('inf'), float('-inf')]:
        # print('scale overflow', scale, scale.max(), x.max(), x.min())

    int_part = torch.floor(input)
    frac_part = input - int_part
    #print(frac_part)
    #print(int_part)
    max_int_scale = 2 ** int(input.max() * 0.9)
    #print(max_int_scale)
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

def custom_int_gelu(x, bw, term):
    raise NotImplementedError

def custom_int_softmax(x, bw, term):
    new_x = x.to(torch.float64)
    # print("x", new_x.max(), new_x.min())
    # x_clamp = torch.clamp(new_x, min = - 20, max = 30)
    x_max = torch.max(new_x, -1, keepdim=True)[0]
    x_norm = new_x - x_max
    x_exp, s = custom_int_exp(x_norm, bw, term)
    if torch.isnan(x_exp).any():
        print('x_exp overflow', x_exp.dtype)
    int_s = 2 ** frac_bits[bw]
    x_exp = (x_exp * int_s).to(torch.int64)
    x_sum = x_exp.sum(dim=-1, keepdim=True)
    if torch.isnan(x_sum).any():
        print('x_sum overflow', x_sum.dtype)
    # print("sum should be", x_exp.sum(dim=-1, keepdim=True), x_sum.max())

    return x_exp.to(torch.float64) / x_sum.to(torch.float64)
    # return frac_div(x_exp, x_sum, bw)
    # return x_exp / x_sum

count = {"1":0}

def custom_int_layernorm(x, w, b, bw):
    eps = 1e-5
    # x_sum_x = torch.tensor(0)
    # x_sum_x2 = torch.tensor(0)
    scale = x.max() * 0.9
    x_1 = x / scale

    int_s = 2 ** frac_bits[bw]
    x_1 = (x * int_s).to(torch.int64)

    N = x_1.shape[-1]
    print(N)
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
    count["1"] += 1
    if count["1"] <= 5:
        print("statistics:", scale, x_sum_x, x_1.mean(dim=-1, keepdim=True, dtype=x_sum_x.dtype))
    invsqrt = 1.0 / (x_sum_x2 - (x_sum_x ** 2) + eps).sqrt()
    # prrint(invsqrt, 1.0 / (x_1.var()))
    ans = w * (x_1 - x_sum_x) * invsqrt + b
    return ans.to(torch.float64)
