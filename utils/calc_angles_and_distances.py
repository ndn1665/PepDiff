import logging
import math
import os
from Bio import PDB

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def has_backbone_atoms(res):
    """
    원자 이름이 N, CA, C를 모두 가지고 있는지 확인
    """
    names = {atom.get_id() for atom in res}
    return {"N", "CA", "C"}.issubset(names)

def calculate_dihedral(vectors):
    """
    네 개의 원자 벡터를 받아서 dihedral 각도를 계산
    계산 중 오류가 발생하면 NaN을 반환
    """
    try:
        return PDB.calc_dihedral(*vectors)
    except (ZeroDivisionError, ValueError, KeyError) as e:
        logging.warning(f"Failed to calculate dihedral angle: {e}")
    return float("nan")

def calculate_angle(angles):
    """
    세 개의 원자 벡터를 받아서 결합 각도(PDB.calc_angle)를 계산
    오류 발생 시 NaN을 반환.
    """
    try:
        return PDB.calc_angle(*angles)
    except (ZeroDivisionError, ValueError, KeyError) as e:
        logging.warning(f"Failed to calculate bond angle: {e}")
        return float("nan")

class GetAngles(object):
    """
    단백질의 백본(주사슬) torsion 각도(phi, psi, omega)와
    결합 각도(C-N-CA, N-CA-C, CA-C-N)만 계산하는 클래스.
    표준 아미노산 잔기(res.id[0] == " " 인 경우)에 대해서만 계산한다.
    """
    def __init__(self, units=None):
        # angles를 어느 단위로 표현할지 설정 (degrees or radians)
        if units is None:
            units = "degrees"
        self.degrees = bool(units[0].lower() == "d")
        if self.degrees:
            logging.debug("Using degrees")
        else:
            logging.debug("Using radians")

        # Bio.PDB 파서 초기화
        self.parser = PDB.PDBParser(QUIET=True)

    def calculate_backbone_torsion(self, chain):
        """
        Calculate backbone torsion angles (phi, psi, omega).
        - phi:    C_(i-1) - N_i - CA_i - C_i
        - psi:    N_i     - CA_i - C_i  - N_(i+1)
        - omega:  CA_(i-1)- C_(i-1)- N_i - CA_i
        """
        torsion_angles = []
        # 표준 아미노산 & 백본 원자 존재하는지 필터
        residues = [
            res for res in chain
            if res.id[0] == " " and has_backbone_atoms(res)
        ]
        num_residues = len(residues)

        for i, res in enumerate(residues):
            phi = psi = omega = float("nan")
            try:
                # phi
                if i > 0:
                    prev_res = residues[i - 1]
                    phi = calculate_dihedral([
                        prev_res['C'].get_vector(),
                        res['N'].get_vector(),
                        res['CA'].get_vector(),
                        res['C'].get_vector()
                    ])
                # psi
                if i < num_residues - 1:
                    next_res = residues[i + 1]
                    psi = calculate_dihedral([
                        res['N'].get_vector(),
                        res['CA'].get_vector(),
                        res['C'].get_vector(),
                        next_res['N'].get_vector()
                    ])
                # omega
                if i > 0:
                    prev_res = residues[i - 1]
                    omega = calculate_dihedral([
                        prev_res['CA'].get_vector(),
                        prev_res['C'].get_vector(),
                        res['N'].get_vector(),
                        res['CA'].get_vector()
                    ])
            except KeyError:
                pass

            # 단위 변환(deg/rad) 적용
            if self.degrees:
                phi    = math.degrees(phi)    if not math.isnan(phi)    else phi
                psi    = math.degrees(psi)    if not math.isnan(psi)    else psi
                omega  = math.degrees(omega)  if not math.isnan(omega)  else omega

            torsion_angles.append([phi, psi, omega])

        return torsion_angles

    def calculate_backbone_bond_angles(self, chain):
        """
        Calculate backbone bond angles (C-N-CA, N-CA-C, CA-C-N).
        - angle1: N - CA - C
        - angle2: C_(i-1) - N_i - CA_i
        - angle3: CA_i - C_i - N_(i+1)
        """
        bond_angles = []
        residues = [
            res for res in chain
            if res.id[0] == " " and has_backbone_atoms(res)
        ]
        num_residues = len(residues)

        for i, res in enumerate(residues):
            angle1 = angle2 = angle3 = float("nan")
            try:
                n  = res['N'].get_vector()
                ca = res['CA'].get_vector()
                c  = res['C'].get_vector()

                # N - CA - C
                angle1 = calculate_angle([n, ca, c])

                # C_(i-1) - N_i - CA_i
                if i > 0:
                    prev_res = residues[i - 1]
                    c_prev = prev_res['C'].get_vector()
                    angle2 = calculate_angle([c_prev, n, ca])

                # CA_i - C_i - N_(i+1)
                if i < num_residues - 1:
                    next_res = residues[i + 1]
                    n_next = next_res['N'].get_vector()
                    angle3 = calculate_angle([ca, c, n_next])

            except KeyError:
                pass

            # 단위 변환(deg/rad) 적용
            if self.degrees:
                angle1 = math.degrees(angle1) if not math.isnan(angle1) else angle1
                angle2 = math.degrees(angle2) if not math.isnan(angle2) else angle2
                angle3 = math.degrees(angle3) if not math.isnan(angle3) else angle3

            bond_angles.append([angle1, angle2, angle3])

        return bond_angles

    def calculate_torsion(self, fn):
        """
        지정된 PDB 파일(fn)에 대해
        - Backbone torsion angles (phi, psi, omega)
        - Backbone bond angles (C-N-CA, N-CA-C, CA-C-N)
        을 계산.
        """
        structure = self.parser.get_structure(
            os.path.splitext(os.path.basename(fn))[0], fn
        )

        backbone_torsions = []
        backbone_bonds = []

        # 파일 내 여러 Model, Chain 가능성 고려
        for model in structure:
            for chain in model:
                torsion = self.calculate_backbone_torsion(chain)
                bonds   = self.calculate_backbone_bond_angles(chain)
                backbone_torsions.extend(torsion)
                backbone_bonds.extend(bonds)

        return backbone_torsions, backbone_bonds


if __name__ == "__main__":
    # 테스트 예시
    calculator = GetAngles(units="degrees")
    backbone_torsions, backbone_bonds = calculator.calculate_torsion('./peptide/6psk_T.pdb')
    
    print("Backbone Torsions (phi, psi, omega):")
    for angles in backbone_torsions:
        print(angles)
    
    print("\nBackbone Bond Angles (C-N-CA, N-CA-C, CA-C-N):")
    for angles in backbone_bonds:
        print(angles)
