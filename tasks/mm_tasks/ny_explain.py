# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional
from argparse import Namespace
import torch
import string
from fairseq import metrics
from fairseq.tasks import register_task

from fairseq import metrics, utils
from tasks.ofa_task import OFAConfig, OFATask
from data.mm_data.ny_explain_dataset import NyExplainDataset
from data.file_dataset import FileDataset
from data import data_utils

logger = logging.getLogger(__name__)


@dataclass
class NyExplainConfig(OFAConfig):
    valid_batch_size: int = field(
        default=20,
        metadata={"help": "valid batch size per step"},
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args for BLUE or CIDEr scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={"help": "prompt_type"},
    )


@register_task("ny_explain", dataclass=NyExplainConfig)
class NyExplainTask(OFATask):
    def __init__(self, cfg: NyExplainConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = NyExplainDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            prompt_type=self.cfg.prompt_type
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_print_samples:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
            
        return model

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        seq_generator = super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs, prefix_allowed_tokens_fn)

        return seq_generator

    def bpe_pertoken_encode(self, toklist):
        return [self.bpe.bpe.decoder[int(t)] if not t.startswith('<') else t
                for t in toklist]

        
    def valid_step(self, sample, model, criterion, **extra_kwargs):

        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        model.eval()
        
        net_output = model(**sample["net_input"])
        # disable non language tokens for ppl
        net_output[0][:, :, 50265:] = -16383. # close to half minimum value

        # this is how you map back to bpe tokenization, but all looks good (e.g.., EOS is there).
        #print(self.bpe_pertoken_encode([self.tgt_dict[t] for t in sample['target'][0].int().cpu()]))

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs_true = lprobs.gather(dim=-1, index=sample['target'].unsqueeze(-1)).squeeze(-1)
        target_mask = (sample['target'] != criterion.padding_idx) * 1.0

        summed_logprobs = (lprobs_true * target_mask).sum(1)
        perplexities = (-summed_logprobs/sample['n_toks_for_ppl']).exp()

        logging_output['ppl_ny'] = perplexities.mean()

        #look to caption for generation
        if self.cfg.eval_print_samples:
            hyps, refs = self._inference(self.sequence_generator, sample, model)

        return loss, sample_size, logging_output


    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        def mean_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs) / len(list(log.get(key, 0) for log in logging_outputs))
            if torch.is_tensor(result):
                result = result.cpu()
            return result
        metrics.log_scalar('ppl_ny', mean_logs('ppl_ny'))

    def _inference(self, generator, sample, model):

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.bpe:
                s = self.bpe.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            decode_tokens = decode(gen_out[i][0]["tokens"])
            hyps.append(decode_tokens.strip())
            refs.append(
                [
                    sent.strip()
                    for sent in decode(
                        utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                        escape_unk=True,  # don't count <unk> as matches to the hypo
                    ).split('&&')
                ]
            )

        print(hyps)
        print(refs)
        quit()
        if self.cfg.eval_print_samples:
            logger.info("example id: {}".format(sample['id'][0]))
            logger.info("example input: "+ [self.bpe.decode(self.src_dict.string(x)).replace('<pad>','') for x in sample['net_input']['src_tokens'] ][0])
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + ' && '.join(refs[0]))
            logger.info("example id: {}".format(sample['id'][1]))
            logger.info("example input: "+ [self.bpe.decode(self.src_dict.string(x)).replace('<pad>','') for x in sample['net_input']['src_tokens'] ][1])
            logger.info("example hypothesis: " + hyps[1])
            logger.info("example reference: " + ' && '.join(refs[1]))

        return hyps, refs
