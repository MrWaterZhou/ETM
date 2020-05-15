import pkuseg
import multiprocessing
from multiprocessing import Process, Queue
import os
import time
import sys


def _proc(seg, in_queue, out_queue):
    # TODO: load seg (json or pickle serialization) in sub_process
    #       to avoid pickle seg online when using start method other
    #       than fork
    while True:

        item = in_queue.get()
        try:
            if item is None:
                return
            idx, line = item
            if not seg.postag:
                output_str = " ".join(seg.cut(line))
            else:
                output_str = " ".join(map(lambda x: "/".join(x), seg.cut(line)))
            out_queue.put((idx, output_str))
        except Exception as e:
            print(item, e)


def _test_multi_proc(
        input_file,
        output_file,
        nthread,
        model_name="default",
        user_dict="default",
        postag=False,
        verbose=False,
):
    alt = multiprocessing.get_start_method() == "spawn"

    times = []
    times.append(time.time())

    if alt:
        seg = None
    else:
        seg = pkuseg.pkuseg(model_name, user_dict, postag)

    times.append(time.time())
    if not os.path.exists(input_file):
        raise Exception("input_file {} does not exist.".format(input_file))
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    times.append(time.time())
    in_queue = Queue()
    out_queue = Queue()
    procs = []
    for _ in range(nthread):
        if alt:
            p = Process(
                target=pkuseg._proc_alt,
                args=(model_name, user_dict, postag, in_queue, out_queue),
            )
        else:
            p = Process(target=_proc, args=(seg, in_queue, out_queue))
        procs.append(p)

    for idx, line in enumerate(lines):
        in_queue.put((idx, line))

    for proc in procs:
        in_queue.put(None)
        proc.start()

    times.append(time.time())
    result = [None] * len(lines)
    for _ in result:
        idx, line = out_queue.get()
        result[idx] = line

    times.append(time.time())
    for p in procs:
        p.join()

    times.append(time.time())
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(result))
    times.append(time.time())

    print("total_time:\t{:.3f}".format(times[-1] - times[0]))

    if verbose:
        time_strs = [
            "load_model",
            "read_file",
            "start_proc",
            "word_seg",
            "join_proc",
            "write_file",
        ]

        if alt:
            times = times[1:]
            time_strs = time_strs[1:]
            time_strs[2] = "load_modal & word_seg"

        for key, value in zip(
                time_strs,
                [end - start for start, end in zip(times[:-1], times[1:])],
        ):
            print("{}:\t{:.3f}".format(key, value))


if __name__ == '__main__':
    _test_multi_proc(sys.argv[1],sys.argv[2],12)
