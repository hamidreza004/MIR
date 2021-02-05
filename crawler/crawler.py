from consts import TOTAL_PAPERS_COUNT
from loader import get_paper, FaultyPageException
import json

queue = ['2981549002', '3105081694', '2950893734']
visited_ids = set(())
seq_number = 0
results = []


def main():
    try:
        while not is_crawling_finished():
            print(seq_number)

            paper_id = get_next_id()

            try:
                paper = get_paper(paper_id)
            except FaultyPageException:
                continue

            print(paper)

            results.append(paper)

            for ref in paper['references']:
                add_id_to_queue(ref)

    finally:
        store_results()
        store_queue()


def is_crawling_finished():
    return len(results) == TOTAL_PAPERS_COUNT


def get_next_id():
    global seq_number

    if seq_number >= len(queue):
        raise Exception('frontier is empty')

    nex_id = queue[seq_number]
    seq_number += 1

    return nex_id


def add_id_to_queue(paper_id):
    if is_paper_visited(paper_id):
        return

    queue.append(paper_id)
    mark_as_visited(paper_id)


def store_results():
    with open('data.json', 'w') as f:
        json.dump(results, f)


def store_queue():
    with open('queue.json', 'w') as f:
        json.dump({"q": queue, "seq": seq_number, "ids": list(visited_ids)}, f)


def is_paper_visited(paper_id):
    return paper_id in visited_ids


def mark_as_visited(paper_id):
    visited_ids.add(paper_id)


if __name__ == '__main__':
    main()
