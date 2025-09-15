#!/usr/bin/env python3
import os
import sys
from Bio import PDB

def rename_chains_in_file(path):
    """
    path: .pdb 확장자가 붙은 interface 파일의 전체 경로
    파일명 형식: <PDBID>.<rec_chain>.<lig_chain>.<l|r>interface.pdb
    rec_chain, lig_chain 중 하나라도 길이가 2 이상인 경우에만
    모든 원래 체인 문자를 A/B로 교체하고 덮어씀
    """
    fname = os.path.basename(path)
    root, ext = os.path.splitext(fname)
    parts = root.split('.')
    if len(parts) != 4 or parts[3] not in ('linterface', 'rinterface'):
        return

    pdb_id, rec_chain_str, lig_chain_str, iface_type = parts

    # 둘 다 한 글자면 스킵
    if len(rec_chain_str) == 1 and len(lig_chain_str) == 1:
        return

    # rec_chain_str 의 모든 문자 → 'A', lig_chain_str 의 모든 문자 → 'B'
    mapping = {c: 'A' for c in rec_chain_str}
    mapping.update({c: 'B' for c in lig_chain_str})

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, path)
    except ValueError as e:
        print(f"[ERROR] {fname}: {e}")
        return

    for model in structure:
        for chain in list(model):
            if chain.id in mapping:
                old = chain.id
                chain.id = mapping[old]

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(path)
    print(f"[RENAMED] {fname}: {mapping}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <interface_folder>")
        sys.exit(1)

    interface_dir = sys.argv[1]
    for fname in os.listdir(interface_dir):
        fullpath = os.path.join(interface_dir, fname)
        if not os.path.isfile(fullpath):
            continue

        # 1) .pdb 확장자 없으면 붙여주기
        if not fname.endswith('.pdb'):
            newname = fname + '.pdb'
            newpath = os.path.join(interface_dir, newname)
            os.rename(fullpath, newpath)
            print(f"[RENAMED EXT] {fname} → {newname}")
        else:
            newpath = fullpath

        # 2) 체인 교체 시도
        rename_chains_in_file(newpath)
