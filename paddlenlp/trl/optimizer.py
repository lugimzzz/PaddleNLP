# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
from paddle import pir
from paddle.base import framework
from paddle.base.framework import Variable, in_dynamic_or_pir_mode, in_pir_mode
from paddle.optimizer.adamw import AdamW


class AdamWMerge(AdamW):
    def __init__(self, reserve_p, alpha, rescale, clip_val, merge_type, tau_ref_state_dict, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        if reserve_p <= 0 or reserve_p >= 1:
            raise ValueError(f"reserve_p must be between 0 and 1, but got {reserve_p}")
        self.reserve_p = reserve_p
        self.alpha = alpha
        self.rescale = rescale
        self.clip_val = clip_val
        self.merge_type = merge_type
        self.eps = eps
        self.tau_ref_state_dict = self.sparsify_state_dict(tau_ref_state_dict)

    def sparsify(self, tensor):
        with paddle.no_grad():
            org_sum = tensor.abs().sum()
            if self.merge_type == "ondare":
                tensor = paddle.nn.functional.dropout(tensor, p=1 - self.reserve_p, training=True)
                new_sum = tensor.abs().sum()
                if self.rescale and org_sum >= 1e-8:
                    tensor *= org_sum / (new_sum + self.eps)
                tensor = paddle.clip(tensor, min=-self.clip_val, max=self.clip_val)
            elif self.merge_type == "onties":
                reserve_num = int(self.reserve_p * tensor.numel())
                if reserve_num < 1:
                    return tensor
                # keep top reserve_num largest elements
                drop_indexs = paddle.argsort(tensor.flatten().abs(), descending=True)[reserve_num:]
                tensor.flatten()[drop_indexs] = 0.0
                new_sum = tensor.abs().sum()
                if self.rescale and org_sum >= 1e-8:
                    tensor *= org_sum / (new_sum)
            else:
                raise ValueError("Unknown merge type {}".format(self.merge_type))

    def sparsify_state_dict(self, state_dict):
        for key in state_dict:
            state_dict[key] = self.sparsify(state_dict[key])
        return state_dict

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param, grad = param_and_grad

        # Whether we should do weight decay for the parameter.
        with_decay = True
        if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(param.name):
            with_decay = False

        moment1 = self._get_accumulator_master(self._moment1_acc_str, param_and_grad[0])
        moment2 = self._get_accumulator_master(self._moment2_acc_str, param_and_grad[0])
        beta1_pow_acc = self._get_accumulator_master(self._beta1_pow_acc_str, param_and_grad[0])
        beta2_pow_acc = self._get_accumulator_master(self._beta2_pow_acc_str, param_and_grad[0])
        tau_ref = self.tau_ref_state_dict[param_and_grad[0].name]
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
        master_weight = self._master_weights[param_and_grad[0].name] if find_master else None
        lr = self._create_param_lr(param_and_grad)
        # create the adamw optimize op
        if in_dynamic_or_pir_mode():
            lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])

            _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
            _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)

            found_inf = self._get_auxiliary_var("found_inf") if in_pir_mode() else None
            self.adamw_merge(
                param_and_grad[0],
                param_and_grad[1],
                lr,
                moment1,
                moment2,
                beta1_pow_acc,
                beta2_pow_acc,
                master_weight,
                found_inf,
                _beta1,
                _beta2,
                self._epsilon,
                lr_ratio_,
                self._weight_decay,
                with_decay,
                find_master,
                tau_ref,
            )
            return None
        else:
            raise NotImplementedError("Not implemented yet.")

    def adamw_merge(
        self,
        param,
        grad,
        learning_rate,
        moment1,
        moment2,
        beta1_pow,
        beta2_pow,
        master_weight,
        skip_update,
        beta1,
        beta2,
        epsilon,
        lr_ratio,
        coeff,
        with_decay,
        multi_precision,
        tau_ref,
    ):
        if skip_update:
            return
        if not with_decay:
            coeff = 0.0
        if not multi_precision:
            master_weight = None
        lr = learning_rate * lr_ratio
        if master_weight is not None:
            p = master_weight
        else:
            p = param
        mom1 = moment1
        mom2 = moment2

        # adam calculation
        mom1 = beta1 * mom1 + (1.0 - beta1) * grad
        mom2 = beta2 * mom2 + (1.0 - beta2) * grad * grad
        denom = mom2.sqrt() / (1.0 - beta2_pow).sqrt() + epsilon
        delta = (moment1 / denom) * (-(lr / (1.0 - beta1_pow)))

        # merge
        p += self.merge(tau_ref, self.sparsify(delta))

        # weight decay
        p *= 1.0 - lr * coeff
        if master_weight is not None:
            master_weight[:] = p
            param[:] = p.astype(param.dtype)
        else:
            param[:] = p
        moment1[:] = mom1
        moment2[:] = mom2
        beta1_pow[:], beta2_pow[:] = beta1 * beta1_pow[:], beta2 * beta2_pow[:]

        return

    def merge(self, tensor1, tensor2):
        if self.merge_type == "ondare":
            tensor = self.alpha * tensor1 + (1 - self.alpha) * tensor2
        elif self.merge_type == "onties":
            tensor = paddle.where(tensor1.abs() > tensor2.abs(), self.alpha * tensor1, (1 - self.alpha) * tensor2)
        else:
            raise ValueError("Unknown merge type {}".format(self.merge_type))
        return tensor


class AdamWPython(AdamW):
    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param, grad = param_and_grad

        # Whether we should do weight decay for the parameter.
        with_decay = True
        if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(param.name):
            with_decay = False

        moment1 = self._get_accumulator_master(self._moment1_acc_str, param_and_grad[0])
        moment2 = self._get_accumulator_master(self._moment2_acc_str, param_and_grad[0])
        beta1_pow_acc = self._get_accumulator_master(self._beta1_pow_acc_str, param_and_grad[0])
        beta2_pow_acc = self._get_accumulator_master(self._beta2_pow_acc_str, param_and_grad[0])
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
        master_weight = self._master_weights[param_and_grad[0].name] if find_master else None
        lr = self._create_param_lr(param_and_grad)
        # create the adamw optimize op
        if in_dynamic_or_pir_mode():
            lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])

            _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
            _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)

            found_inf = self._get_auxiliary_var("found_inf") if in_pir_mode() else None
            self.adamw_python(
                param_and_grad[0],
                param_and_grad[1],
                lr,
                moment1,
                moment2,
                beta1_pow_acc,
                beta2_pow_acc,
                master_weight,
                found_inf,
                _beta1,
                _beta2,
                self._epsilon,
                lr_ratio_,
                self._weight_decay,
                with_decay,
                find_master,
            )
            return None
        else:
            raise NotImplementedError("Not implemented yet.")

    def adamw_python(
        self,
        param,
        grad,
        learning_rate,
        moment1,
        moment2,
        beta1_pow,
        beta2_pow,
        master_weight,
        skip_update,
        beta1,
        beta2,
        epsilon,
        lr_ratio,
        coeff,
        with_decay,
        multi_precision,
    ):
        if skip_update:
            return
        if not with_decay:
            coeff = 0.0
        if not multi_precision:
            master_weight = None
        lr = learning_rate * lr_ratio
        if master_weight is not None:
            p = master_weight
        else:
            p = param
        p *= 1.0 - lr * coeff
        mom1 = moment1
        mom2 = moment2

        mom1 = beta1 * mom1 + (1.0 - beta1) * grad
        mom2 = beta2 * mom2 + (1.0 - beta2) * grad * grad
        denom = mom2.sqrt() / (1.0 - beta2_pow).sqrt() + epsilon
        p += (moment1 / denom) * (-(lr / (1.0 - beta1_pow)))
        if master_weight is not None:
            master_weight[:] = p
            param[:] = p.astype(param.dtype)
        else:
            param[:] = p
        moment1[:] = mom1
        moment2[:] = mom2
        beta1_pow[:], beta2_pow[:] = beta1 * beta1_pow[:], beta2 * beta2_pow[:]
        # 看看怎么更新
        return
