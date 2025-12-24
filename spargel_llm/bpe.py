import time
from collections import Counter
from multiprocessing import Pipe, Process, cpu_count
from typing import Iterable, Optional, Sequence

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


def _remove_length_one_samples(samples: list[list[int]]):
    pos = 0
    for sample in samples:
        if len(sample) >= 2:
            samples[pos] = sample
            pos += 1
    del samples[pos:]


def _count_pairs_in_seqs(
    counter: Counter[tuple[int, int]], seqs: Iterable[Sequence[int]]
):
    for seq in seqs:
        counter.update((seq[i], seq[i + 1]) for i in range(len(seq) - 1))


def _apply_merge(seq: list[int], id1: int, id2: int, new_id: int):
    """
    Find all adjacent pairs of (id1, id2) in a sequence and replace them with new_id.
    """

    # no pairs: nothing to do
    if len(seq) <= 1:
        return

    # We do the merging in place so no extra memory allocation is needed.
    # There won't be any RAW (Read After Write) since we always have pos <= cursor.
    cursor = 0
    pos = 0
    while cursor < len(seq) - 1:
        if seq[cursor] == id1 and seq[cursor + 1] == id2:
            seq[pos] = new_id
            cursor += 2
        else:
            seq[pos] = seq[cursor]
            cursor += 1
        pos += 1

    # last one
    if cursor == len(seq) - 1:
        seq[pos] = seq[cursor]
        pos += 1

    del seq[pos:]


def _apply_merge_to_chunk(
    samples_chunk: list[list[int]], id1: int, id2: int, new_id: int
):
    for sample in samples_chunk:
        _apply_merge(sample, id1, id2, new_id)


def _bpe_expand_worker_func(samples: list[list[int]], conn):
    counter = Counter()
    while True:
        signal = conn.recv()
        match signal:
            case 0:
                break
            case 1:
                _remove_length_one_samples(samples)

                counter.clear()
                _count_pairs_in_seqs(counter, samples)
                conn.send(counter)
                counter.clear()

                id1, id2, new_id = conn.recv()
                if new_id == 0:
                    continue
                _apply_merge_to_chunk(samples, id1, id2, new_id)
            case _:
                raise ValueError(f"unknown signal: ${signal}")

    conn.close()


def bpe_expand(
    words: list[bytes],
    samples: list[list[int]],
    count: int,
    *,
    num_processes: Optional[int] = None,
):

    if count == 0:
        return

    if num_processes is None:
        num_processes = cpu_count()

    if num_processes == 1:
        bpe_expand_simple(words, samples, count)
        return

    n_samples = len(samples)
    base_size = n_samples // num_processes
    remainder = n_samples % num_processes

    processes: list[Process] = []
    conns = []

    for i in range(num_processes):
        start = i * base_size + min(i, remainder)
        end = start + base_size + (1 if i < remainder else 0)

        parent_conn, child_conn = Pipe()
        conns.append(parent_conn)
        process = Process(
            target=_bpe_expand_worker_func, args=(samples[start:end], child_conn)
        )
        processes.append(process)
        process.start()

    total_counter = Counter()
    for _ in range(count):
        t_start = time.perf_counter()

        for conn in conns:
            conn.send(1)

        total_counter.clear()
        for conn in conns:
            counter = conn.recv()
            total_counter.update(counter)

        most_common = total_counter.most_common(1)
        if len(most_common) == 0:
            print("No more pairs. Stopping.")
            for conn in conns:
                conn.send((0, 0, 0))
            break
        total_counter.clear()

        new_id = len(words)
        (id1, id2), freq = most_common[0]
        new_word = words[id1] + words[id2]
        words.append(new_word)

        for conn in conns:
            conn.send((id1, id2, new_id))

        t_end = time.perf_counter()

        print(
            f"word {new_id}: {id1}+{id2} freq={freq} \t{new_word} ({repr(new_word.decode(errors="ignore"))}) {t_end-t_start:.6f}s"
        )

    for conn in conns:
        conn.send(0)

    for process in processes:
        process.join()


def bpe_expand_simple(words: list[bytes], samples: list[list[int]], count: int):
    """
    Find frequent pairs and replace them with new words.

    Args:
        words: the list of words, id == index; new words will be appended to it.
        samples: from these we find pairs (will be changed)
        count: number of new words to find (will stop early if no more pairs)
    """

    counter = Counter()
    for _ in range(count):

        t_start = time.perf_counter()

        # remove seqs with len=1
        pos = 0
        for sample in samples:
            if len(sample) >= 2:
                samples[pos] = sample
                pos += 1
        del samples[pos:]

        new_id = len(words)

        counter.clear()
        _count_pairs_in_seqs(counter, samples)
        most_common = counter.most_common(1)
        if len(most_common) == 0:
            print("No more pairs. Stopping.")
            break
        counter.clear()

        (id1, id2), freq = most_common[0]

        if freq == 0:
            print("No more pairs. Stopping.")
            break

        new_word = words[id1] + words[id2]

        words.append(new_word)

        for sample in samples:
            _apply_merge(sample, id1, id2, new_id)

        t_end = time.perf_counter()

        print(
            f"word {new_id}: {id1}+{id2} freq={freq} \t{new_word} ({repr(new_word.decode(errors="ignore"))}) {t_end-t_start:.6f}s"
        )
