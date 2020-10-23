from molecule import Molecule
from helpers import qed, penalized_logp, sas_smiles, similarity

class QEDMolEnv(Molecule):

    def __init__(self, atom_types, init_mol='C', max_steps=45):
        super().__init__(
            atom_types=atom_types,
            init_mol=init_mol,
            max_steps=max_steps,
            record_path=True
        )

    def _reward(self):
        return qed(self.state)



class PenalizedLogpEnv(Molecule):

    def __init__(self, atom_types, init_mol='C', max_steps=45):
        super().__init__(
            atom_types=atom_types,
            init_mol=init_mol,
            max_steps=max_steps,
            record_path=True
        )


    def _reward(self):
        return penalized_logp(self.state)


class BenchmarkEnv(Molecule):

    def __init__(self, atom_types, init_mol='C', max_steps=45):
        super().__init__(
            atom_types=atom_types,
            init_mol=init_mol,
            max_steps=max_steps,
            record_path=True
        )

    def _reward(self):
        return sas_smiles(self.state)


class SimVsQedEnv(Molecule):

    def __init__(self, target_mol, fingerprint_provider, w, 
                atom_types={'C', 'O', 'N', 'Cl'}, init_mol='C', max_steps=45):
        super().__init__(
            atom_types=atom_types,
            init_mol=init_mol,
            max_steps=max_steps,
            record_path=True
        )
        
        self.mfp = fingerprint_provider
        self.target_fp = self.mfp.get_fingerprint_raw(target_mol)
        self.w = w

    def _reward(self):
        current_fp = self.mfp.get_fingerprint_raw(self.state)
        return self.w * qed(self.state) + (1-self.w) * similarity(self.target_fp, current_fp)