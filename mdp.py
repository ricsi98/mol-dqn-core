from molecule import Molecule
from helpers import qed, penalized_logp, sas_smiles

class QEDMolEnv(Molecule):

    def __init__(self, atom_types, init_mol='C', max_steps=45):
        super().__init__(
            atom_types=atom_types,
            init_mol=init_mol,
            max_steps=max_steps
        )

    def _reward(self):
        return qed(self.state)



class PenalizedLogpEnv(Molecule):

    def __init__(self, atom_types, init_mol='C', max_steps=45):
        super().__init__(
            atom_types=atom_types,
            init_mol=init_mol,
            max_steps=max_steps
        )


    def _reward(self):
        return penalized_logp(self.state)


class BenchmarkEnv(Molecule):

    def __init__(self, atom_types, init_mol='C', max_steps=45):
        super().__init__(
            atom_types=atom_types,
            init_mol=init_mol,
            max_steps=max_steps
        )

    def _reward(self):
        return sas_smiles(self.state)