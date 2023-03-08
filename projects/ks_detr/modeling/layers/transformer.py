import copy
import warnings
from typing import List
import torch
import torch.nn as nn

from detrex.layers import BaseTransformerLayer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class KSBaseTransformerLayer(BaseTransformerLayer):

    def __init__(
            self,
            attn: List[nn.Module],
            ffn: nn.Module,
            norm: nn.Module,
            operation_order: tuple = None,
    ):
        super().__init__(attn=attn, ffn=ffn, norm=norm, operation_order=operation_order)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor = None,
            value: torch.Tensor = None,
            query_pos: torch.Tensor = None,
            key_pos: torch.Tensor = None,
            attn_masks: List[torch.Tensor] = None,
            query_key_padding_mask: torch.Tensor = None,
            key_padding_mask: torch.Tensor = None,
            **kwargs,
    ):
        """Forward function for `BaseTransformerLayer`.

        **kwargs contains the specific arguments of attentions.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (torch.Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (torch.Tensor): The position embedding for `query`.
                Default: None.
            key_pos (torch.Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (torch.Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        intermediate_output = {}

        for layer in self.operation_order:
            # if layer == "self_attn":
            if layer.startswith("self_attn"):
                temp_key = temp_value = query  # torch.Size([2, 10850, 256])
                query, *self_attn_out = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1

                assert isinstance(query, list), 'returned query should be a list'
                identity = query

                # update intermediate_output
                if self_attn_out:
                    assert isinstance(self_attn_out[0], dict)
                    intermediate_output.update(self_attn_out[0])

            elif layer == "norm":

                assert isinstance(query, list)
                for k in range(len(query)):
                    query[k] = self.norms[norm_index](query[k])

                norm_index += 1

            elif layer.startswith("cross_attn"):
                # self_attn in decoder layer should have one attn module, not multi-attention module.
                assert isinstance(query, list) and len(query) == 1

                assert query == identity

                query, *cross_attn_out = self.attentions[attn_index](
                    query[0],
                    key,
                    value,
                    identity[0] if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                # update intermediate_output with cross_attn_out
                if cross_attn_out:
                    assert isinstance(cross_attn_out[0], dict)
                    intermediate_output.update(cross_attn_out[0])

                attn_index += 1
                identity = query

            elif layer == "ffn":
                for k in range(len(query)):
                    query[k] = self.ffns[ffn_index](query[k], identity[k] if self.pre_norm else None)
                ffn_index += 1

        assert isinstance(query, list)
        intermediate_output['feat'] = query[0]

        if len(query) >= 2:
            # Here is for handling the attn_map for self-attn in encoder.
            # Note: decoder cross-attention has been handled inside cross-attention function (
            #  saved in intermediate_output ['ca_attn_map_logits_t', 'ca_attn_map_logits',
            # 'sa_attn_map_logits']
            intermediate_output['feat_t'] = query[1:]
            if 'attn_map_logits' in intermediate_output:
                assert isinstance(intermediate_output['attn_map_logits'], (list, tuple))
                if len(intermediate_output['attn_map_logits']) >= 2:
                    intermediate_output['attn_map_logits'] = list(intermediate_output['attn_map_logits'])
                    intermediate_output['attn_map_logits_t'] = intermediate_output['attn_map_logits'][1:]
                    intermediate_output['attn_map_logits'] = intermediate_output['attn_map_logits'][0]

        query = query[0]

        return query, intermediate_output


class KSTransformerLayerSequence(nn.Module):
    """DETR typically use identical layers in its encoder or decoder. KS-DETR enables using different layers in the
    encoder or decoder.
    """

    def __init__(
            self,
            encoder_decoder_layer_list=None,

            transformer_layers=None,
            num_layers=None,
    ):
        super(KSTransformerLayerSequence, self).__init__()

        # Initialize layers first from encoder_decoder_layer_list, if encoder_decoder_layer_list is None,
        # then go to the pre-defined way of layer initialization.
        if encoder_decoder_layer_list is not None:
            self.layers = nn.ModuleList()
            self.num_layers = 0
            for l_conf in encoder_decoder_layer_list:
                encoder_layer, num_l = l_conf
                assert num_l > 0
                # nn.ModuleList
                self.layers.extend(_get_clones(encoder_layer, num_l))
                self.num_layers += num_l
        else:
            self.num_layers = num_layers
            self.layers = nn.ModuleList()
            if isinstance(transformer_layers, nn.Module):
                for _ in range(num_layers):
                    self.layers.append(copy.deepcopy(transformer_layers))
            else:
                assert isinstance(transformer_layers, list) and len(transformer_layers) == num_layers

    def forward(self):
        """Forward function of `TransformerLayerSequence`. The users should inherit
        `TransformerLayerSequence` and implemente their own forward function.
        """
        raise NotImplementedError()


