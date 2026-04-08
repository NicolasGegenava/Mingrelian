import json
import sys

def dump_nb(in_file, out_file):
    with open(in_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                f.write(f"\n# MARKDOWN: {''.join(cell['source'])}\n")
            elif cell['cell_type'] == 'code':
                f.write(f"\n# CODE:\n{''.join(cell['source'])}\n")

if __name__ == '__main__':
    dump_nb(sys.argv[1], sys.argv[2])
