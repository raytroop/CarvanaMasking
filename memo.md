train_set: 5088 sample
train: (1280, 1918, 3)
train_masks: (1280, 1918, 3)



torch.nn.Module:
- apply(fn): Applies fn `recursively` to every submodule (as returned by .children()) as well as self.
    def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.data.fill_(1.0)
        print(m.weight)
    net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    net.apply(init_weights)

    ###
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)

    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)

    Sequential(
    (0): Linear(in_features=2, out_features=2, bias=True)
    (1): Linear(in_features=2, out_features=2, bias=True)
    )

- children()与modules()的区别
    children()与modules()都是返回网络模型里的组成元素，但是children()返回的是最外层的元素，modules()返回的是所有的元素，包括不同级别的子元素。
    https://discuss.pytorch.org/t/module-children-vs-module-modules/4551

    用list举例就是：

    a=[1,2,[3,4]]
    children返回
    1,2，[3，4]
    modules返回
    [1,2,[3,4]], 1, 2, [3,4], 3, 4
