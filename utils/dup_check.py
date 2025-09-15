"""
train–test 데이터에서
· train 세트와 **같은 서열**(길이 ≥ 3) 을 가진 모든 test PDB ID를 모아
  ➜ duplicate_test_seqs.txt 에 저장한다.

파일 형식 (FASTA-like):
    >test_PDB_ID
    SEQUENCE
"""

"""

현재 test dataset에는 161개의 sequence가 train dataset과 중복된다.
그중 길이 3이 넘는 sequence는 38개이다.
test_dataset에 7466개의 PDB ID가 있는 점을 고려했을때 38개의 중복 서열은 상당히 적은 편이다.
"""

import pickle
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────
#  1) 데이터 로드
# ──────────────────────────────
train_path = "sequence_angles_train.pkl"
test_path  = "sequence_angles_test.pkl"

with open(train_path, "rb") as f:
    train_ds = pickle.load(f)
with open(test_path, "rb") as f:
    test_ds = pickle.load(f)

# ──────────────────────────────
#  2) 서열 → {PDB ID 집합} 매핑
# ──────────────────────────────
train_seq2pdbs = defaultdict(set)
for seq, pdb_id in zip(train_ds["sequences"], train_ds["pdb_file"]):
    train_seq2pdbs[seq].add(pdb_id)

test_seq2pdbs = defaultdict(set)
for seq, pdb_id in zip(test_ds["sequences"], test_ds["pdb_file"]):
    test_seq2pdbs[seq].add(pdb_id)

# ──────────────────────────────
#  3) 교집합 서열(길이 ≥ 3)에 해당하는 test-쪽 PDB ID 수집
# ──────────────────────────────
dup_test_ids = set()
for seq in train_seq2pdbs.keys() & test_seq2pdbs.keys():
    if len(seq) >= 3:                             # 길이 조건
        dup_test_ids.update(test_seq2pdbs[seq])

dup_test_ids = sorted(dup_test_ids)               # 리스트 정렬

# ──────────────────────────────
#  4) test PDB ID → 서열 매핑
# ──────────────────────────────
test_pdb2seq = dict(zip(test_ds["pdb_file"], test_ds["sequences"]))

# ──────────────────────────────
#  5) 파일로 저장
# ──────────────────────────────
out_path = Path("duplicate_test_seqs.txt")
with out_path.open("w", encoding="utf-8") as fout:
    fout.write(f"# 길이 ≥ 3 중복 서열을 가진 test PDB ID 개수: {len(dup_test_ids)}\n\n")
    for pdb_id in dup_test_ids:
        seq = test_pdb2seq[pdb_id]
        fout.write(f">{pdb_id}\n{seq}\n\n")

print(f"[INFO] 결과 저장 완료 → {out_path.resolve()}")
