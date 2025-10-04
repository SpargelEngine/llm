import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Sequence

from .meta import ai_marker

# Priority of merge.
Rank = int

MAX_RANK: Rank = 1_000_000
MAX_OFFSET: int = 1_000_000_000


def byte_pair_merge(ranks: dict[bytes, Rank], piece: bytes) -> list[int]:
    """
    Perform merges with the give ranks.

    Note: rank means priority for a merge, i.e. the lower the rank is, the more urgent the merge is.

    The input bytes are merged into segments.

    Args:
        ranks: a dictionary specifying ranks for some byte sequences
               Note: it is assume that `ranks[s] <= ranks[t]` if `s` is a prefix of `t`, and
                     every key in `ranks` can be written as the sum of another two keys in `ranks`.
        piece: the input byte sequence

    Return: the start position of each segment
    """
    # every item represents a part of `piece` with rank
    parts: list[tuple[int, Rank]] = []

    # index of the the merge point (in `parts`) with minimal rank, i.e. maximal merge priority
    min_rank: tuple[int, Rank] = (MAX_OFFSET, MAX_RANK)

    # iterate over the adjacent bytes
    for i in range(len(piece) - 1):
        # if the byte-pair does not exist in `ranks`, assign inf to rank
        rank = ranks.get(piece[i : i + 2], MAX_RANK)
        parts.append((i, rank))
    parts.append((len(piece) - 1, MAX_RANK))
    # add a virtual merge point at the end
    parts.append((len(piece), MAX_RANK))

    min_rank = parts[min(range(len(parts)), key=lambda i: parts[i][1])]

    # sanity check
    assert len(parts) == len(piece) + 1

    # get the rank of byte-pair formed by merge points `i`, `i+1`, `i+2`
    # note: this is called when `i` and `i+1` will be merged, or when `i+1` and `i+2` will be merged
    def get_rank(i: int) -> Rank:
        if i + 3 < len(parts):
            p = piece[parts[i][0] : parts[i + 3][0]]
            return ranks.get(p, MAX_RANK)
        else:
            return MAX_RANK

    # loop condition: there are byte-pairs that can be merged
    while not min_rank[1] == MAX_RANK:
        # the offset of the merge point with minimal rank
        i = min_rank[0]

        # we need to recompute the rank at the previous byte if there is one
        if i > 0:
            # only the rank is modified
            parts[i - 1] = (parts[i - 1][0], get_rank(i - 1))
        parts[i] = (parts[i][0], get_rank(i))
        # remove the next merge point, as it has been merged with `i`
        parts.pop(i + 1)

        # find the next merge point with minimal rank
        min_index = min(range(len(parts)), key=lambda i: parts[i][1])
        min_rank = (min_index, parts[min_index][1])

    return [segment[0] for segment in parts[:-1]]


@ai_marker(human_checked=True)
def _count_pairs_in_chunk(chunk: list[Sequence[int]]) -> Counter[tuple[int, int]]:

    def pairwise[T](seq: Sequence[T]):
        for i in range(len(seq) - 1):
            yield seq[i], seq[i + 1]

    counter = Counter[tuple[int, int]]()
    for sample in chunk:
        counter += Counter(pairwise(sample))
    return counter


@ai_marker(human_checked=True)
def find_most_frequent_pair(
    samples: Sequence[Sequence[int]],
    *,
    num_processes: Optional[int] = None,
) -> tuple[int, int, int]:
    """
    Args:
        samples: from all these we count the frequencies of pairs
        num_processes: number of processes to use (default: number of CPU cores)
        chunk_size: number of sequences to process in each chunk (default: 1000)

    Return:
        id1, id2, freq: the most frequent pair and its frequency (return freq = 0 when no pairs)
    """
    samples_list = list(samples)

    if len(samples_list) == 0:
        return 0, 0, 0

    # Use all available CPUs if not specified
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # For small datasets, use sequential processing to avoid overhead
    if num_processes == 1 or len(samples_list) < num_processes:
        counter = _count_pairs_in_chunk(samples_list)
    else:
        # Split samples into chunks
        chunk_size = len(samples_list) // num_processes

        chunks = [
            samples_list[i : i + chunk_size]
            for i in range(0, len(samples_list), chunk_size)
        ]

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            chunk_counters = list(executor.map(_count_pairs_in_chunk, chunks))

        # Merge all counters
        counter = Counter[tuple[int, int]]()
        for chunk_counter in chunk_counters:
            counter += chunk_counter

    # Find the most common pair
    most_common = counter.most_common(1)
    if len(most_common) == 0:
        return 0, 0, 0
    else:
        pair, cnt = most_common[0]
        return *pair, cnt
