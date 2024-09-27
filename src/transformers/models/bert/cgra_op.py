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
    xmin = torch.amin(x, dim=-1, keepdim=True)
    scale = (((xmax - xmin)*0.9).clamp(min=1e-5) / maxq)
    zero = -xmin
    q = torch.clamp(torch.round((x + zero) / scale), 0, maxq)

    return q, scale, zero

# def asym_quantize(x: torch.Tensor, bits: int):
#     minq, maxq = get_minq_maxq(bits=bits, sym=False)
#     xmax = torch.amax(x, dim=-1, keepdim=True)
#     xmin = torch.zeros_like(xmax)
#     # print("xmax, xmin", xmax, xmin, x.max(), x.min(), maxq)
#     # print("sub", xmax - xmin, "max", (xmax - xmin).max())
#     # print("clamp", ((xmax - xmin)*0.9).clamp(min=1e-5))
#     print("clamp result", ((xmax - xmin)*0.9).max())
#     scale = (xmax - xmin)*0.9
#     print("scale in this, ", scale, scale.max())
#     zero = torch.zeros_like(xmax)
#     q = torch.clamp(torch.round((x + zero) / scale), -xmax, xmax)

#     return q, scale, zero

def asym_dequantize(q, scale, zero):
    return q * scale - zero

def custom_int_gelu(x, bw, term):
    raise NotImplementedError

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
    if(ans >= 2 **(bw-1)).any():
        print('multiplication overflow')
    ans[ans >= 2 ** (bw - 1)] = (2 ** (bw - 1)) - 1
    # ans[ans >= 2 ** (bw - 1)] = (2 ** (bw - 1)) - 1
    # print(ans)
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
    print("call int exp")
    print("x", x.max(), x.min())
    fp_x = x.to(torch.float64)
    input = fp_x * torch.tensor(1.442695)

    int_part = torch.floor(input)
    frac_part = input - int_part
    #print(frac_part)
    #print(int_part)
    max_int_scale = 2 ** int(scale.max() * 0.9)
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

def custom_int_softmax(x, bw, term):
    # new_x = x.to(torch.float64)
    # print("x", new_x.max(), new_x.min())
    # x_clamp = torch.clamp(new_x, min = - 20, max = 30)
    # x_max = torch.max(x_clamp, -1, keepdim=True)[0]
    # x_norm = x_clamp - x_max
    x_max = torch.max(x, -1, keepdim=True)[0]
    x = x - x_max
    # x_exp = custom_int_exp(x.to(dtype=torch.float64), bw, term)
    x_exp, s = custom_int_exp(x.to(dtype=torch.float64), bw, term)
    if torch.isnan(x_exp).any():
        print('x_exp overflow', x_exp.dtype)
    int_s = 2 ** frac_bits[bw]
    x_exp = (x_exp * int_s).to(torch.int64)
    x_sum = x_exp.sum(dim=-1, keepdim=True)
    if torch.isnan(x_sum).any() or x_sum.max() >= int_s:
        print('x_sum overflow', x_sum.dtype)
    # print("sum should be", x_exp.sum(dim=-1, keepdim=True), x_sum.max())

    return x_exp.to(torch.float64) / x_sum.to(torch.float64)
    # return frac_div(x_exp, x_sum, bw)
    # return x_exp / x_sum
