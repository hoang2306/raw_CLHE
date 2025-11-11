from torch import nn
import torch
import torch.nn.functional as F


class MyLinear(nn.Linear):
    # pass
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

        # print("init weight:")
        # print(self.weight)
        # asdfasdfadf

        # nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MyNormalLinear(nn.Linear):
    # pass
    def reset_parameters(self) -> None:
        # nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MyPReLU(nn.Module):
    __constants__ = ["num_parameters"]
    num_parameters: int

    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        super().__init__()

        # use alpha instead of weight
        self.alpha = nn.parameter.Parameter(
            torch.empty(num_parameters, **factory_kwargs).fill_(init)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.prelu(input, self.alpha)

    def extra_repr(self) -> str:
        return "num_parameters={}".format(self.num_parameters)


class Lambda(nn.Module):
    def __init__(self, func) -> None:
        super().__init__()

        self.func = func

    def forward(self, x):
        return self.func(x)


def create_act(name=None):
    if name == "softmax":
        return nn.Softmax(dim=-1)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return Lambda(lambda x: x)
    else:
        raise Exception()


# class MyLinear(nn.Linear):
#     def reset_parameters(self) -> None:
#         nn.init.xavier_normal_(self.weight)
#         nn.init.zeros_(self.bias)


def get_activation(activation):
    if activation == "prelu":
        return MyPReLU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation is None or activation == "none":
        return torch.nn.Identity()
    else:
        raise NotImplementedError()


class MyMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        units_list,
        activation,
        drop_rate,
        bn,
        output_activation,
        output_drop_rate,
        output_bn,
        ln=False,
        output_ln=False,
    ):
        super().__init__()

        layers = []
        units_list = [in_channels] + units_list  # Add in_channels to the list of units

        for i in range(len(units_list) - 1):
            layers.append(
                MyLinear(units_list[i], units_list[i + 1])
            )  # Create a linear layer
            # layers.append(MyNormalLinear(units_list[i], units_list[i+1]))  # Create a linear layer

            if i < len(units_list) - 2:
                if bn:
                    layers.append(
                        nn.BatchNorm1d(units_list[i + 1])
                    )  # Add a batch normalization layer

                if ln:
                    layers.append(nn.LayerNorm(units_list[i + 1]))

                layers.append(
                    get_activation(activation)
                )  # Add the PReLU activation function
                layers.append(nn.Dropout(drop_rate))
            else:
                if output_bn:
                    layers.append(nn.BatchNorm1d(units_list[i + 1]))

                if output_ln:
                    layers.append(nn.LayerNorm(units_list[i + 1]))

                layers.append(
                    get_activation(output_activation)
                )  # Add the PReLU activation function
                layers.append(nn.Dropout(output_drop_rate))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

class Sampled_Transformer(nn.Module):
    def __init__(
        self,
        in_units,
        att_units,
        out_units,
        ff_units_list,
        num_heads=1,
        drop_rate=0.0,
        att_h_drop_rate=0.0,
        att_drop_rate=0.0,
        ff_drop_rate=0.0,
        output_activation=None,
        output_drop_rate=0.0,
        output_ln=False,
        att_residual=True,
        ff_residual=True,
        ln=True,
    ) -> None:
        super().__init__()

        self.q_linear = MyLinear(in_units, att_units)
        self.k_linear = MyLinear(in_units, att_units)
        # self.v_linear = MyLinear(in_units, out_units)
        self.ff_units_list = ff_units_list

        # ff_units_list = [out_units] + ff_units_list

        if ln:
            self.ln = nn.LayerNorm(out_units)
        else:
            self.ln = None

        self.att_h_dropout = nn.Dropout(att_h_drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.att_dropout = nn.Dropout(att_drop_rate)

        if len(ff_units_list) == 0:
            self.ff = None
        else:
            self.ff = MyMLP(
                out_units,
                ff_units_list,
                activation="prelu",
                drop_rate=drop_rate,
                bn=True,
                output_activation="prelu",
                output_drop_rate=0.0,
                output_bn=True,
                # ln=True,
                # ln=True,
                # output_ln=True
            )

        self.output_activation = get_activation(output_activation)
        self.output_dropout = nn.Dropout(ff_drop_rate)
        # self.output_bn = nn.BatchNorm1d(ff_units_list[-1]) if output_bn else None
        if len(ff_units_list) > 0:
            self.output_ln = nn.LayerNorm(ff_units_list[-1]) if output_ln else None

        self.att_residual = att_residual
        self.ff_residual = ff_residual

        self.num_heads = num_heads

    def forward(self, q, k, return_all=False):
        Q = self.q_linear(q)
        K = self.k_linear(k)
        V = k

        Q_ = torch.concat(Q.split(Q.size(-1) // self.num_heads, dim=-1), dim=0)
        K_ = torch.concat(K.split(K.size(-1) // self.num_heads, dim=-1), dim=0)
        V_ = torch.concat(V.split(V.size(-1) // self.num_heads, dim=-1), dim=0)

        sim = Q_ @ K_.transpose(-2, -1)
        sim /= Q_.size(-1) ** 0.5

        sim_logits = sim

        sim = F.softmax(sim, dim=-1)

        sim = self.att_dropout(sim)

        att_h = sim @ V_

        att_h = torch.concat(att_h.split(q.size(0), dim=0), dim=-1)

        no_residual_att_h = att_h

        if self.att_residual:
            # att_h = att_h + q
            att_h = att_h * 0.1 + q * 0.9
            # att_h = q

        if self.ln is not None:
            att_h = self.ln(att_h)

        att_h = self.att_h_dropout(att_h)

        if self.ff is None:
            ff_h = att_h
        else:
            # ff_h = self.ff(att_h)

            ff_h = self.ff(att_h.squeeze(1)).unsqueeze(1)

            if self.ff_residual:
                ff_h = ff_h + att_h

            if self.output_ln:
                ff_h = self.output_ln(ff_h)

            ff_h = self.output_activation(ff_h)
            ff_h = self.output_dropout(ff_h)

        if return_all:
            return ff_h, no_residual_att_h, sim_logits
        else:
            return ff_h