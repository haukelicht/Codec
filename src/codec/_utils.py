"""
Internal utilities: data structures, token-ID mappings, and text-processing helpers.
"""
import math
import re
import string

import torch


# ---------------------------------------------------------------------------
# Token IDs for bracket markers per model family.
# Each model maps the marker *symbol* to the list of token IDs that represent
# it in that model's vocabulary (multiple IDs exist because some tokenisers
# produce both a space-prefixed and a non-prefixed variant).
# ---------------------------------------------------------------------------
MODEL2BRACKET_IDS: dict[str, dict[str, list[int]]] = {
    "m2m": {
        "[": [542],
        "]": [11355],
    },
    "nllb": {
        "[": [709],    # token: _[
        "]": [10109],  # token: _]
        "(": [104],
        ")": [14229],
    },
    "mbart": {
        "[": [378],
        "]": [10114],
    },
}

BRACKET_IDS: set[int] = {709, 248415, 10109, 248414}

PUNC_LIST: set[str] = set(string.punctuation)
PUNC_LIST.remove("'")

CHINESE_PUNC = (
    '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟'
    '〰〾〿–—\u2018\u2019‛\u201c\u201d„‟…‧﹏'
)


# ---------------------------------------------------------------------------
# Search data structures
# ---------------------------------------------------------------------------

class Candidate:
    """
    Accumulates the top-n translation hypotheses produced by the constrained
    search.

    The hypotheses are maintained in a min-heap so that the worst-scoring
    candidate can be evicted efficiently when a better one is found.

    Attributes:
        text_ids: Token IDs of the current best single hypothesis (used as a
            lower-bound during branch pruning).
        score: Log-probability of the current worst hypothesis in the heap
            (i.e. the eviction threshold).
        accumulate_scores: Per-step cumulative log-probabilities for the
            current lower-bound hypothesis, used by the heuristic future-cost
            estimate.
        min_heap: List of ``(score, -count, token_ids, acc_scores,
            [open_pos, close_pos])`` tuples maintained as a min-heap.  The
            ``-count`` tiebreaker ensures a stable ordering when scores are
            equal.
        count: Monotonically increasing counter used as a heap tiebreaker.
        search_tree: Root ``Node`` of the visualisation tree (only populated
            when ``save_visualization=True``).
        open_bracket_position: Token index of ``[`` in the best hypothesis.
        close_bracket_position: Token index of ``]`` in the best hypothesis.
    """

    def __init__(
        self,
        text_ids: torch.LongTensor | None,
        score: torch.FloatTensor,
        accumulate_scores: torch.FloatTensor | None = None,
    ):
        self.text_ids = text_ids
        self.score = score
        self.flag = False
        self.count = 0.0
        self.max_position = 0
        self.accumulate_scores = accumulate_scores
        self.search_tree = None
        self.close_bracket_position = -1
        self.open_bracket_position = -1
        self.min_heap: list = []

    def to_cuda(self, device: int) -> None:
        """Move tensor attributes to *device*."""
        if self.text_ids is not None:
            self.text_ids = self.text_ids.to(device)
        if self.score is not None:
            self.score = self.score.to(device)

    def update_smallest_candidate(self) -> None:
        """Sync scalar attributes from the current heap root (worst candidate)."""
        self.score = self.min_heap[0][0]
        self.text_ids = self.min_heap[0][2]
        self.accumulate_scores = self.min_heap[0][3]


class Node:
    """
    A node in the visualisation search tree.

    Each node represents one decoding step (one token choice).  The tree is
    only constructed when ``save_visualization=True`` is passed to the search.

    Attributes:
        text: Token ID chosen at this step.
        log_prob: Step log-probability ``log P(token | context)``.
        acc_log_prob: Cumulative log-probability from the start of decoding up
            to and including this step.
        upperbound: The best-known score threshold at the time this node was
            expanded (used to decide whether the node was worth exploring).
        child: Child nodes (next decoding steps branching from here).
        level: Depth in the tree (root = 0).
    """

    def __init__(
        self,
        text: int | None = None,
        log_prob: float | None = None,
        acc_log_prob: float | None = None,
        upperbound: float | None = None,
    ):
        self.text = text
        self.log_prob = log_prob
        self.acc_log_prob = acc_log_prob
        self.child: list["Node"] = []
        self.level = 0
        self.upperbound = upperbound

    def add_child_node(self, node: "Node") -> None:
        self.child.append(node)
        node.level = self.level + 1

    def __repr__(self) -> str:
        return "{} - {}: {}/{}".format(
            self.level,
            self.text,
            round(self.log_prob, 2),
            round(self.acc_log_prob, 2),
        )


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------

def print_tree(root_node: Node | None) -> None:
    """Recursively print the search tree rooted at *root_node*."""
    if not root_node:
        return
    print(root_node)
    for node in root_node.child:
        print_tree(node)


# ---------------------------------------------------------------------------
# Text-processing helpers
# ---------------------------------------------------------------------------

def check_punctuation(text: str) -> bool:
    """Return True if *text* is a Chinese or ASCII punctuation character."""
    return text in CHINESE_PUNC or text in PUNC_LIST


def preprocess(mt, md, text, is_tokenized: bool = False) -> str:
    """
    Moses-tokenise *text* and re-detokenise each token individually so that
    the output is a space-separated string of surface forms.

    Parameters:
        mt: ``MosesTokenizer`` instance.
        md: ``MosesDetokenizer`` instance.
        text: Input text (raw string or pre-tokenised list when
            ``is_tokenized=True``).
        is_tokenized: If True, *text* is already a list of tokens.
    """
    tokenized = mt.tokenize(text) if not is_tokenized else text
    return " ".join(md.tokenize([tok]) for tok in tokenized)


def preprocess2(text, org_text: str, mt, md, is_tokenized: bool = False, lang=None) -> str:
    """
    Like :func:`preprocess` but aligns the output against *org_text* so that
    the original casing / spacing is preserved where possible.
    """
    tokenized = mt.tokenize(text) if not is_tokenized else text
    org_text = re.sub(" +", " ", org_text)
    parts = [md.detokenize([tok]) for tok in tokenized]
    p1 = 0
    tokenized_parts = []
    for part in parts:
        if org_text[p1:].startswith(part):
            tokenized_parts.append(f" {part} " if part in PUNC_LIST else org_text[p1: p1 + len(part)])
            p1 += len(part)
        elif org_text[p1 + 1:].startswith(part):
            tokenized_parts.append(f" {part} " if part in PUNC_LIST else org_text[p1: p1 + 1 + len(part)])
            p1 += 1 + len(part)
        else:
            print(text)
            print(org_text)
            print("Fail")
            break
    return re.sub(" +", " ", "".join(tokenized_parts)).strip()


def post_process(text: str, org_text: str) -> str:
    """
    Re-insert bracket markers from *text* into *org_text*, preserving the
    original surface form of every non-bracket token.

    This is used to map decoded output (which may contain bracket tokens
    produced by the model's tokeniser) back onto the original detokenised
    string.
    """
    x_pointer = 0
    org_x_pointer = 0
    new_str: list[str] = []
    if org_text.strip() == "":
        return org_text
    while x_pointer < len(text) or org_x_pointer < len(org_text):
        if org_x_pointer == len(org_text):
            assert text[x_pointer] in (" ", "[", "]")
            new_str.append(text[x_pointer])
            x_pointer += 1
            continue
        if x_pointer == len(text):
            new_str.append(org_text[org_x_pointer])
            org_x_pointer += 1
            continue
        if org_text[org_x_pointer] == text[x_pointer]:
            new_str.append(org_text[org_x_pointer])
            x_pointer += 1
            org_x_pointer += 1
        else:
            if text[x_pointer] in ("[", "]"):
                new_str.append(text[x_pointer])
                x_pointer += 1
            else:
                assert (
                    org_text[org_x_pointer] == " " or text[x_pointer] == " "
                ), f"\n {org_text} \n {text}"
                if text[x_pointer] == " " and org_text[org_x_pointer] != " ":
                    x_pointer += 1
                elif text[x_pointer] != " " and org_text[org_x_pointer] == " ":
                    new_str.append(org_text[org_x_pointer])
                    org_x_pointer += 1
    return "".join(new_str)


def tokenize_non_whitespace(template_text: str, tokenizer) -> list[int]:
    """
    Tokenise *template_text* while suppressing spurious leading ``▁`` tokens
    that SentencePiece inserts at word boundaries inside a template string.

    This is needed for Chinese and other scripts where the template must be
    tokenised without treating every character as a new word.

    Parameters:
        template_text: Detokenised target-language template string.
        tokenizer: HuggingFace tokenizer with a SentencePiece back-end.

    Returns:
        List of token IDs (no special tokens).
    """
    tokens: list[str] = []
    for i, word in enumerate(template_text.split(" ")):
        if i == 0:
            tokens.extend(tokenizer.tokenize(word))
        else:
            # Strip leading ▁ from subsequent words so positions stay aligned
            # with whitespace-separated surface forms.
            pieces = [p for p in tokenizer.tokenize(word) if p != "▁"]
            tokens.extend(p.replace("▁", "") for p in pieces)
    return [tokenizer.convert_tokens_to_ids(t) for t in tokens]


def compute_number_combination(
    n: int,
    num_brackets: int,
    num_choices_per_bracket: int = 2,
) -> int:
    """
    Estimate the size of the search space.

    Computes ``C(n + num_brackets, num_brackets) * choices^num_brackets``,
    which approximates the number of ways to insert *num_brackets* bracket
    pairs into a sequence of *n* tokens when each bracket has
    *num_choices_per_bracket* token variants.
    """
    return math.comb(n + num_brackets, num_brackets) * (num_choices_per_bracket ** num_brackets)


@torch.no_grad()
def enc_dec_scoring(
    input_ids: torch.LongTensor,
    target_ids: torch.LongTensor,
    model,
    attention_mask: torch.LongTensor | None = None,
) -> torch.FloatTensor:
    """
    Score *target_ids* given *input_ids* using an encoder-decoder *model*.

    Parameters:
        input_ids: Source token IDs, shape ``(1, src_len)``.
        target_ids: Target token IDs (labels), shape ``(1, tgt_len)``.
        model: HuggingFace encoder-decoder model.
        attention_mask: Optional attention mask for *input_ids*.

    Returns:
        Per-token log-probabilities, shape ``(1, tgt_len, vocab_size)``.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=target_ids,
        return_dict=True,
    )
    return outputs.logits
