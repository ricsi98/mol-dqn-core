from molecule import Molecule
from helpers import qed

class QEDMolEnv(Molecule):

    def __init__(self, atom_types, init_mol='C', max_steps=45):
        super().__init__(
            atom_types=atom_types,
            init_mol=init_mol,
            max_steps=max_steps
        )

    def _reward(self):
        qed(self.state)