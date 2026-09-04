import torch

# 继承自PyTorch Optimizer。有所有优化器共有的基础设施，并初始化
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # step 1: 基本校验（lr>=0、0<=beta<1 等）
        # 把超参打包进 defaults 交给基类
        defaults = dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)

        # 调用父类 torch.optim.Optimizer 的 __init__，让父类帮我们把「模型参数」和「优化器超参数」初始化好
        super().__init__(params, defaults)

    # 根据当前各个参数的梯度 param.grad，真正更新模型参数
    # 用@torch.no_grad()是因为后面你会直接更新参数 p，不希望这些更新操作进入 autograd 计算图
    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        # closure 是一个可选函数，用来重新执行 forward、计算 loss 并 backward。
        # 普通 AdamW 训练通常不用 closure，因此 step() 一般返回 None；
        # 如果传入 closure，则调用 closure 得到 loss，完成参数更新后返回这个 loss。
        # 这个函数的存在是因为有些优化器需要重新计算 loss + gradient 来进行优化
        # 这个函数相当于是一个重新计算 loss + gradient 按钮
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历 self.param_groups，再遍历 group["params"] 里每个 p.grad 不为 None 的参数：
        for group in self.param_groups:
            # step 2: 取出这一组的超参 lr、(beta1, beta2)、eps、weight_decay
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                # 如果 p.grad 为 None，说明这个参数在这次反向传播中没有梯度，不需要更新
                if p.grad is None:
                    continue

                # 根据优化算法更新 p
                # step 3: 用 self.state[p] 维护该参数的持久状态，也就是上面理论上提到过的
                # t：step；一阶矩 m_t：exp_avg（意思是 梯度的指数移动平均）；二阶矩 v_t：exp_avg_sq（意思是 梯度平方的指数移动平均）
                # 然后在第一次需要初始化 state，注意Optimizer 父类的初始化逻辑会帮你创建 self.state：
                state = self.state[p]

                # 步数计数 t=0；一阶矩 m 与二阶矩 v（形状同 p，全 0）
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                # step 4: 读梯度 grad = p.grad（自问：需要 grad.data 还是 grad 就行？）
                # grad = p.grad 即可；grad.data 会绕过 autograd 的计算图追踪，可能导致隐蔽的梯度问题
                # 因此现代 PyTorch 一般不推荐使用。优化器更新参数时通常用 @torch.no_grad() 显式关闭梯度追踪。
                grad = p.grad # g_t

                # step 5: 步数 +1，更新有偏一阶矩 m 与有偏二阶矩 v（各自的指数滑动平均）
                state["step"] += 1 # t

                # 使用原地更新
                # state["exp_avg"] = beta1 * state["exp_avg"] + (1 - beta1) * grad 
                # state["exp_avg_sq"] = beta2 * state["exp_avg_sq"] + (1 - beta2) * (grad ** 2) 
                state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1) # m_t
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # v_t
                
                # step 6: 偏差校正 —— 想清楚是校正 m、v 本身，还是把校正吸收进等效步长 alpha_t（两种写法在数学上等价；handout 用后者）
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                alpha_t = lr * (bias_correction2 ** 0.5) / bias_correction1 # 吸收偏差校正后的等效步长

                # step 7: 解耦权重衰减：另起一步  p <- p - lr * weight_decay * p
                # 注意它不经过 sqrt(v) 的自适应缩放 —— 这正是 AdamW 与 "Adam+L2" 的分水岭
                # 注意这里也要原地更改
                # p = p - lr * weight_decay * p
                p.mul_(1 - lr * weight_decay)

                # step 8: 自适应更新：用 m / (sqrt(v) + eps) 的方向、以 alpha_t 为步长更新 p（原地更改）
                # p = p - alpha_t * (state["exp_avg"] / (state["exp_avg_sq"] ** 0.5 + eps))
                p.add_(state["exp_avg"] / (state["exp_avg_sq"] ** 0.5 + eps), alpha = - alpha_t)

        return loss  # 若用到了 closure，记得返回