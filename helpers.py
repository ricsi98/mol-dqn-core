# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Tools for manipulating graphs and converting from atom and pair features."""

from __future__ import absolute_import, division, print_function

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
import rdkit.Chem.QED as QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from SA_Score import sascorer
from rdkit.Chem import Draw


def atom_valences(atom_types):
    """Creates a list of valences corresponding to atom_types.

    Note that this is not a count of valence electrons, but a count of the
    maximum number of bonds each element will make. For example, passing
    atom_types ['C', 'H', 'O'] will return [4, 1, 2].

    Args:
      atom_types: List of string atom types, e.g. ['C', 'H', 'O'].

    Returns:
      List of integer atom valences.
    """
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type)))
        for atom_type in atom_types
    ]


def get_scaffold(mol):
    """Computes the Bemis-Murcko scaffold for a molecule.

    Args:
      mol: RDKit Mol.

    Returns:
      String scaffold SMILES.
    """
    return Chem.MolToSmiles(
        MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=True)


def contains_scaffold(mol, scaffold):
    """Returns whether mol contains the given scaffold.

    NOTE: This is more advanced than simply computing scaffold equality (i.e.
    scaffold(mol_a) == scaffold(mol_b)). This method allows the target scaffold to
    be a subset of the (possibly larger) scaffold in mol.

    Args:
      mol: RDKit Mol.
      scaffold: String scaffold SMILES.

    Returns:
      Boolean whether scaffold is found in mol.
    """
    pattern = Chem.MolFromSmiles(scaffold)
    matches = mol.GetSubstructMatches(pattern)
    return bool(matches)


def get_largest_ring_size(molecule):
    """Calculates the largest ring size in the molecule.

    Refactored from
    https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py

    Args:
      molecule: Chem.Mol. A molecule.

    Returns:
      Integer. The largest ring size.
    """
    cycle_list = molecule.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def penalized_logp(molecule):
    """Calculates the penalized logP of a molecule.

    Refactored from
    https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
    See Junction Tree Variational Autoencoder for Molecular Graph Generation
    https://arxiv.org/pdf/1802.04364.pdf
    Section 3.2
    Penalized logP is defined as:
    y(m) = logP(m) - SA(m) - cycle(m)
    y(m) is the penalized logP,
    logP(m) is the logP of a molecule,
    SA(m) is the synthetic accessibility score,
    cycle(m) is the largest ring size minus by six in the molecule.

    Args:
    molecule: Chem.Mol. A molecule.

    Returns:
    Float. The penalized logP value.

    """
    molecule = Chem.MolFromSmiles(molecule)
    if molecule is None: return 0
    try:
        log_p = Descriptors.MolLogP(molecule)
        sas_score = sascorer.calculateScore(molecule)
        largest_ring_size = get_largest_ring_size(molecule)
        cycle_score = max(largest_ring_size - 6, 0)
        return log_p - sas_score - cycle_score
    except:
        return 0


def sajat(molecule):
    m = Chem.MolFromSmiles(molecule)
    if m is None: return 0
    try:
        sas = sascorer.calculateScore(m)
        qed = QED.qed(m)
        return 5 * qed - sas
    except:
        return 0


def penalized_logp_batch(molecules):
    return [penalized_logp(m) for m in molecules]


def cos_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def similarity_qed_reward(molecule, base_molecule_fingerprint, w, hparams):
    molecule_fingerprint = get_fingerprint(molecule, hparams)
    sim = cos_similarity(molecule_fingerprint, base_molecule_fingerprint)
    qed = QED.qed(Chem.MolFromSmiles(molecule))
    return sim * w + (1-w) * qed


def logp(mol):
    return Descriptors.MolLogP(Chem.MolFromSmiles(mol))

def sas(mol):
    return sascorer.calculateScore(mol)

def sas_smiles(mol):
    mol = Chem.MolFromSmiles(mol)
    return sascorer.calculateScore(mol)

def qed(mol):
	return QED.qed(Chem.MolFromSmiles(mol))


def logp_mol(mol):
	return Descriptors.MolLogP(mol)

def sas_mol(mol):
    return sascorer.calculateScore(mol)

def qed_mol(mol):
	return QED.qed(mol)


def mol2svg(mol):
	return Draw.MolsToGridImage([mol], useSVG=True)

def smiles2mol(mol):
	return Chem.MolFromSmiles(mol)


class MorganFingerprintProvider():
    
    def __init__(self, fp_size, fp_radius):
        self.hparams = {
            'fingerprint_length': fp_size,
            'fingerprint_radius': fp_radius
        }

    def get_fingerprint(self, mol):
        return get_fingerprint(mol, self.hparams)

    def get_fingerprint_raw(self, mol):
        return get_fingerprint_raw(mol, self.hparams)

def get_fingerprint(smiles, hparams):
    """Get Morgan Fingerprint of a specific SMILES string.
    Args:
    smiles: String. The SMILES string of the molecule.
    Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
    """
    if smiles is None:
        return np.zeros((hparams['fingerprint_length'],))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros((hparams['fingerprint_length'],))
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, hparams['fingerprint_radius'], hparams['fingerprint_length'])
    arr = np.zeros((1,))
    # ConvertToNumpyArray takes ~ 0.19 ms, while
    # np.asarray takes ~ 4.69 ms
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr

def get_fingerprint_raw(smiles, hparams):
    """Get Morgan Fingerprint of a specific SMILES string.
    Args:
    smiles: String. The SMILES string of the molecule.
    Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
    """
    if smiles is None:
        None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, hparams['fingerprint_radius'], hparams['fingerprint_length'])
    return fingerprint



def similarity(fp1, fp2):
    if fp1 is None or fp2 is None: return 0
    return DataStructs.TanimotoSimilarity(fp1, fp2)