"""
Command-line interface for the Codec span-annotation translation pipeline.

Reads a JSON file of examples, runs the bracket-constrained search for each,
and writes results back to a JSON file.  The model is loaded **once** and
reused across all examples.

Input JSON format
-----------------
Each element is a dict with the following keys:

``template`` (str)
    The target-language machine-translated sentence, without bracket markers.

``flag`` (int)
    Processing status.  ``1`` means the example should be decoded; ``0`` or
    ``-1`` means it is skipped (e.g. no spans were found in the source).

``text_to_decode`` (list of dict)
    One entry per span-marker pair to insert.  Each entry has:

    - ``"text"`` (str): Source sentence with bracket markers for this span.
    - ``"candidates"`` (list of int): Allowed template token indices for the
      opening marker.  Empty list = unconstrained search for this span.
    - ``"right_candidates"`` (list of list of int, optional): Per-opening-
      position allowed closing positions.  Element *i* corresponds to
      ``candidates[i]``.

``src_entities`` (list of str)
    Source-side entity strings (one per span); carried through to the output
    unchanged.

Output JSON format
------------------
Each result element mirrors the input and adds:

``tgt_lang`` (list of list of str)
    Per-span list of decoded target sentences for the top-*n_best* hypotheses.

``tgt_entities`` (list of list of str)
    Per-span list of decoded entity strings extracted from between the markers.

``score`` (list of list of float)
    Per-span list of hypothesis log-probabilities.

Usage
-----
.. code-block:: bash

    python -m src.codec.cli \\
        --model_name_or_path ychenNLP/nllb-200-distilled-1.3B-easyproject \\
        --tokenizer_path facebook/nllb-200-distilled-1.3B \\
        --input_path data/input.json \\
        --output_path data/output.json \\
        --tgt_lang deu_Latn
"""
import argparse
import json
import pickle
import os
from timeit import default_timer as timer

import torch
from tqdm import tqdm

from .codec import Codec
from ._utils import tokenize_non_whitespace


def _split(lst: list, n: int):
    """Split *lst* into *n* roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))


def main(args: argparse.Namespace) -> None:
    codec = Codec(
        model_name_or_path=args.model_name_or_path,
        tokenizer_path=args.tokenizer_path,
        src_lang=args.src_lang,
        mt_name=args.mt_name,
        max_length=args.max_length,
    )

    with open(args.input_path) as f:
        input_data = json.load(f)

    if args.shard_num is not None:
        input_data = list(_split(input_data, args.total_shard))[args.shard_num]

    print(f"Decoding {len(input_data)} examples …")

    acc_time = 0.0
    results = []
    decoded_token_ids = []

    for i in tqdm(range(len(input_data))):
        record = input_data[i]
        template_text: str = record["template"]
        flag: int = record["flag"]

        # Skip examples that were marked as not needing decoding.
        if flag == 0 or flag == -1:
            results.append({
                "idx": i,
                "src_lang": [],
                "template": template_text,
                "tgt_lang": [],
                "score": [],
                "flag": flag,
            })
            decoded_token_ids.append(None)
            continue

        # --- Tokenise template (shared across all spans) ---------------
        if args.tgt_lang.startswith("zh"):
            pre_template_ids = tokenize_non_whitespace(template_text, codec.tokenizer)
        else:
            pre_template_ids = codec.tokenizer(
                template_text,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=False,
            ).input_ids

        template_ids = torch.LongTensor(
            [codec.tokenizer.eos_token_id, codec.tokenizer.lang_code_to_id[args.tgt_lang]]
            + pre_template_ids
            + [codec.tokenizer.eos_token_id]
        )

        # --- Resolve whitespace-only position constraint ---------------
        whitespace_positions: set[int] | None = None
        if args.white_space_only:
            whitespace_positions = set()
            for idx, tok in enumerate(codec.tokenizer.tokenize(template_text)):
                if tok.startswith("▁"):
                    whitespace_positions.add(idx + 2)

        # --- Per-span decoding -----------------------------------------
        input_text_list: list[str] = []
        output_text_list: list[list[str]] = []
        output_text_ids_list: list[torch.Tensor] = []
        output_acc_log_prob_list: list[list] = []
        score_list: list[list[float]] = []
        all_tgt_entities: list[list[str]] = []
        all_marker_positions: list[list[list[int]]] = []

        left_marker = "(" if args.use_round_marker else "["
        right_marker = ")" if args.use_round_marker else "]"
        bracket_stack = [right_marker, left_marker]

        for text_to_decode in record["text_to_decode"]:
            src_text_with_marker: str = text_to_decode["text"]
            raw_candidates: list[int] = text_to_decode.get("candidates", [])
            raw_right_candidates: list[list[int]] | None = text_to_decode.get("right_candidates")

            # Resolve opening-position constraint.
            possible_opening_positions: set[int] | None
            if args.not_use_candidate or len(raw_candidates) == 0:
                possible_opening_positions = None
            else:
                possible_opening_positions = set(raw_candidates)

            # Resolve closing-position constraint.
            possible_closing_positions: dict[int, set[int]] | None = None
            if not args.not_use_right_candidate and raw_right_candidates is not None:
                possible_closing_positions = {
                    k: set(raw_right_candidates[idx])
                    for idx, k in enumerate(raw_candidates)
                }

            # Apply whitespace-only override.
            if whitespace_positions:
                if possible_opening_positions:
                    possible_opening_positions = possible_opening_positions & whitespace_positions
                else:
                    possible_opening_positions = whitespace_positions
                possible_closing_positions = {k: whitespace_positions for k in possible_opening_positions}

            # Tokenise source for this span.
            input_ids = codec.tokenizer(
                src_text_with_marker,
                return_tensors="pt",
                max_length=args.max_length,
                truncation=True,
            ).input_ids.to(codec.device)

            # Run search via low-level API so we can pass pre-computed template_ids.
            from ._decoding_argument import BracketConstraintDecodingArgument
            from ._generation import generate

            decode_args = BracketConstraintDecodingArgument(
                template_ids=template_ids,
                bracket_stack=bracket_stack,
                template_pointer=1,
                model_name=args.mt_name,
                future_steps=args.future_steps,
                search_mode=args.search_mode,
                batch_size=args.batch_size,
                n_best=args.n_best,
                left_marker=left_marker,
                right_marker=right_marker,
                possible_opening_positions=possible_opening_positions,
                possible_closing_positions=possible_closing_positions,
                save_visualization=args.save_visualization,
            )

            start = timer()
            outputs = generate(
                self=codec.model,
                inputs=input_ids,
                decoding_argument=decode_args.arguments,
                forced_bos_token_id=codec.tokenizer.lang_code_to_id[args.tgt_lang],
                max_length=args.max_length,
                num_beams=args.num_beams,
                length_penalty=0,
            )
            acc_time += timer() - start

            input_text_list.append(src_text_with_marker)

            if not outputs.min_heap:
                # No valid hypothesis found; fall back to the unmodified template.
                output_text_ids_list.append(template_ids.unsqueeze(0).cpu())
                all_marker_positions.append([[0, 0]])
                output_text_list.append([template_text])
                all_tgt_entities.append([""])
                score_list.append([-float("inf")])
            else:
                outputs.min_heap.sort(reverse=True)
                cand_scores: list[float] = []
                cand_text_ids: list[torch.Tensor] = []
                cand_acc_log_probs: list = []
                cand_entities: list[str] = []
                marker_positions_per_span: list[list[int]] = []

                for score, _neg_count, token_ids, acc_log_probs, (lpos, rpos) in outputs.min_heap:
                    marker_positions_per_span.append([lpos, rpos])
                    if lpos < rpos - 1:
                        entity_ids = token_ids[0][lpos + 1: rpos]
                        cand_entities.append(codec.tokenizer.decode(entity_ids))
                    else:
                        cand_entities.append("")
                    cand_scores.append(score)
                    cand_text_ids.append(token_ids)
                    cand_acc_log_probs.append(acc_log_probs)

                all_cand_ids = torch.cat(cand_text_ids, dim=0)
                output_text_list.append(
                    codec.tokenizer.batch_decode(all_cand_ids, skip_special_tokens=True)
                )
                output_text_ids_list.append(all_cand_ids.cpu())
                output_acc_log_prob_list.append(cand_acc_log_probs)
                score_list.append(cand_scores)
                all_tgt_entities.append(cand_entities)
                all_marker_positions.append(marker_positions_per_span)

        decoded_token_ids.append([output_text_ids_list, all_marker_positions])
        results.append({
            "idx": i,
            "src_lang": input_text_list,
            "src_entities": record.get("src_entities", []),
            "template": template_text,
            "tgt_lang": output_text_list,
            "tgt_entities": all_tgt_entities,
            "score": score_list,
            "flag": flag,
        })

        # Optionally save the search-tree visualisation.
        if args.save_visualization and args.search_tree_path:
            out_tree_dir = os.path.join(args.search_tree_path, "trees")
            os.makedirs(out_tree_dir, exist_ok=True)
            with open(os.path.join(out_tree_dir, f"tree_{i}.pkl"), "wb") as f:
                pickle.dump(outputs.search_tree, f)

    print(f"Total running time: {acc_time:.2f}s")
    print(f"Avg time per sample: {acc_time / len(input_data):.2f}s")

    if args.time_log:
        with open(args.time_log, "w") as f:
            f.write(f"Total running time: {acc_time}\nAvg time per sample: {acc_time / len(input_data)}\n")

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        torch.save(decoded_token_ids, args.output_path.replace(".json", ".pt"))
    else:
        print(results)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Codec bracket-constrained span-annotation translation."
    )
    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="HuggingFace model ID or local path for the MT model.")
    p.add_argument("--tokenizer_path", type=str, default=None,
                   help="HuggingFace tokenizer ID or local path.  Defaults to "
                        "--model_name_or_path.")
    p.add_argument("--input_path", type=str, required=True,
                   help="Path to the input JSON file.")
    p.add_argument("--output_path", type=str, default=None,
                   help="Path for the output JSON file.  Prints to stdout when "
                        "omitted.")
    p.add_argument("--src_lang", type=str, default="eng_Latn",
                   help="FLORES-200 source language code (default: eng_Latn).")
    p.add_argument("--tgt_lang", type=str, required=True,
                   help="FLORES-200 target language code (e.g. deu_Latn).")
    p.add_argument("--max_length", type=int, default=1024,
                   help="Maximum sequence length for tokenisation and generation.")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Number of search branches expanded in parallel on the GPU.")
    p.add_argument("--shard_num", type=int, default=None,
                   help="0-based index of the shard to process.  Use together "
                        "with --total_shard to split a large input across "
                        "multiple processes / GPUs.")
    p.add_argument("--total_shard", type=int, default=10,
                   help="Total number of shards (used with --shard_num).")
    p.add_argument("--search_mode", choices=[0, 1, 2], type=int, default=0,
                   help="Search algorithm: 0 = fast heuristic (default), "
                        "1 = slow heuristic (lower GPU memory), "
                        "2 = constrained beam search.")
    p.add_argument("--n_best", type=int, default=5,
                   help="Number of top hypotheses to return per span.")
    p.add_argument("--num_beams", type=int, default=5,
                   help="Beam size (only used for search_mode=2).")
    p.add_argument("--future_steps", type=int, default=-1,
                   help="Look-ahead horizon for the heuristic upper bound.  "
                        "-1 uses the full remaining template (default).")
    p.add_argument("--not_use_candidate", type=int, choices=[0, 1], default=0,
                   help="1 = ignore pre-computed opening position candidates "
                        "(unconstrained search).")
    p.add_argument("--not_use_right_candidate", action="store_true",
                   help="Ignore pre-computed closing position candidates.")
    p.add_argument("--white_space_only", action="store_true",
                   help="Restrict marker insertion to token positions that start "
                        "a new word (SentencePiece ▁ prefix).")
    p.add_argument("--use_round_marker", action="store_true",
                   help="Use round brackets ( ) instead of square brackets [ ].")
    p.add_argument("--mt_name", type=str, default="nllb",
                   help="MT model family (nllb | mbart | m2m).  Controls which "
                        "token IDs are used for bracket markers.")
    p.add_argument("--save_visualization", action="store_true",
                   help="Build and save the search tree (for debugging).")
    p.add_argument("--search_tree_path", type=str, default=None,
                   help="Directory to write search-tree pickle files when "
                        "--save_visualization is set.")
    p.add_argument("--time_log", type=str, default=None,
                   help="Path to write a plain-text timing summary.")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_name_or_path
    args.not_use_candidate = args.not_use_candidate == 1

    print(f"not_use_candidate: {args.not_use_candidate}")
    main(args)
