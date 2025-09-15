"""
Wrapper to load in a PDB file as a pyrosetta pose, relax it, and write back
out the relaxed pose as a PDB file. Also prints out the energy before and
after relaxation.
"""
import argparse
import pyrosetta
from pyrosetta import rosetta

# PyRosetta 초기화 (에러 메시지 억제)
pyrosetta.init("-ignore_unrecognized_res True -mute all")

def relax_pdb(fname: str, out_fname: str):
    """
    Relax a pose and print the energy before and after.
    
    Args:
        fname (str): Input PDB file path.
        out_fname (str): Output PDB file path for the relaxed structure.
    """
    try:
        # 포즈 로드 및 점수 함수 생성
        pose = rosetta.core.import_pose.pose_from_file(fname)
        scorefxn = pyrosetta.get_fa_scorefxn()

        # Relax 전 에너지 계산 및 출력
        initial_energy = scorefxn(pose)
        print(f"Energy before relax: {initial_energy:.4f}")

        # FastRelax 프로토콜 설정 및 적용
        relax = rosetta.protocols.relax.FastRelax()
        relax.set_scorefxn(scorefxn)
        relax.apply(pose)
        
        # Relax 후 에너지 계산 및 출력
        final_energy = scorefxn(pose)
        print(f"Energy after relax: {final_energy:.4f}")

        # Relax된 포즈를 PDB 파일로 저장
        pose.dump_pdb(out_fname)

    except Exception as e:
        print(f"Error processing {fname}: {e}")
        # 오류 발생 시 빈 파일 생성 방지를 위해 종료
        exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Basic CLI parser for single file processing."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_pdb", type=str, help="Input .pdb file to be relaxed.")
    parser.add_argument("output_pdb", type=str, help="Output path for the relaxed .pdb file.")
    return parser


def main():
    """Run script."""
    args = build_parser().parse_args()
    relax_pdb(args.input_pdb, args.output_pdb)


if __name__ == "__main__":
    main()
