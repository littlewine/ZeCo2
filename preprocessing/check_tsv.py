"""
Check the tsv file if it is ready for colbert indexing.
"""
from argparse import ArgumentParser

def main(args):
    nr_errors = 0
    f = open(args.tsv_path, "r")
    probl_lines = set()
    for line_idx, line in enumerate(f):
        line_parts = line.strip().split('\t')
        # Assign pid, passage
        try:
            pid, passage, *other = line_parts
        except ValueError:
            nr_errors +=1
            print(f"ValueError @ line splitting\n"
                  f"Line: {line_parts}\n"
                  f"line_idx: {line_idx}\n")
            probl_lines.add(line_idx)
            continue
            #TODO: fix the file here?

        # Colbert assertions
        if not len(passage) >= 1:
            nr_errors+=1
            print(f"assert len(passage) >= 1"
                  f"Line: {line_parts}"
                  f"line_idx: {line_idx}")

        if not (pid == 'id' or int(pid) == line_idx):
            nr_errors+=1
            print(f"assert pid == 'id' or int(pid) == line_idx"
                  f"Line: {line_parts}"
                  f"line_idx: {line_idx}"
                  f"pid: {pid}")

    print(f"\n\n*********\n\n"
          f"Finished with {nr_errors} errors")
    print(probl_lines)



if __name__ == "__main__":
    parser = ArgumentParser(description="preprocessing queries.")

    parser.add_argument('--tsv_path', required=True, type=str)

    args = parser.parse_args()
    main(args)
