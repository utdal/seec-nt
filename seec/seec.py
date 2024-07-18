from dca.dca_class import dca
from .genetics import GeneticsTools
import numpy as np
from Bio import SeqIO
from dca.dca_functions import return_Hamiltonian, create_numerical_MSA
from numba import jit


@jit(nopython=True)
def getProbDistribution(
    aa_seq: np.array, eij: np.array, localfield: np.array, mut_pos: int
) -> np.array:
    not_selected = np.arange(aa_seq.shape[0], dtype=np.int32)
    not_selected = not_selected[not_selected != mut_pos]
    output_probs = np.zeros(eij.shape[-1], dtype=eij.dtype)

    for position in not_selected:
        if position > mut_pos:
            output_probs += eij[mut_pos, position, :, aa_seq[position]]
        else:
            output_probs += eij[position, mut_pos, aa_seq[position], :]

    output_probs += localfield[:, mut_pos]
    output_probs = np.exp(output_probs)
    return output_probs / np.sum(output_probs)


class SEECnt:
    def __init__(self, msa: str, symboldict: str = "-ACDEFGHIKLMNPQRSTVWY") -> None:
        # Build mfDCA model based on input MSA
        self.dca_params = dca(msa, stype=symboldict)
        self.dca_params.mean_field()
        # Build genetic code helper class
        self.genetics = GeneticsTools(symboldict=symboldict)

    def verifySequences(self, numeric_aa: np.array, numeric_nt: np.array) -> bool:
        if len(numeric_aa) != len(numeric_nt):
            print(
                f"AA Length: {len(numeric_aa)} does not match NT length: {len(numeric_nt)}"
            )
            return False
        elif len(numeric_aa) != self.dca_params.couplings.shape[0]:
            print(
                f"Input AA seq does not match MSA sequence length ({len(numeric_aa)} != {self.dca_params.couplings.shape[0]})"
            )
            return False
        else:
            return True

    def mutationStep(self, aa_seq: np.array, nt_seq: np.array, temp: float) -> list:
        # choose uniformly from all non-gap positions in aa_seq
        non_gap_positions = np.arange(len(aa_seq))[
            aa_seq != self.dca_params.symboldict["-"]
        ]
        pos_choice = np.random.choice(non_gap_positions)
        # get conditional prob distribution
        prob_dist = getProbDistribution(
            aa_seq,
            self.dca_params.couplings / temp,
            self.dca_params.localfields / temp,
            pos_choice,
        )

        # run 100 samples
        samples = np.random.multinomial(1, prob_dist, size=100)

        # filter out all samples that are gaps
        no_gap_samples = samples[samples[:, self.dca_params.symboldict["-"]] != 1]
        if len(no_gap_samples) == 0:
            # unable to find non-gap mutation.
            return aa_seq, nt_seq

        # find only valid codon mutations
        proposed_aa = np.argmax(no_gap_samples, axis=1)
        current_nt = [nt_seq[pos_choice].item()] * len(proposed_aa)
        check_mutations = map(self.genetics.isMutable, current_nt, proposed_aa)
        valid_mutations = [int(res[1]) for res in check_mutations if res[0]]
        if len(valid_mutations) == 0:
            # could not find a codon-adjacent mutation
            return aa_seq, nt_seq

        # create new sequence with first mutation in list
        new_nt = valid_mutations[0]
        new_aa = self.genetics.codonToAA[new_nt]
        new_aa_seq = aa_seq.copy()
        new_nt_seq = nt_seq.copy()
        new_nt_seq[pos_choice] = new_nt
        new_aa_seq[pos_choice] = new_aa
        return new_aa_seq, new_nt_seq

    def evolveSequence(
        self,
        input_AASeq: str,
        input_NTSeq: str,
        num_steps: int,
        selection_temp: float = 1.0,
    ) -> list:
        # Get sequence info encoded and verified
        aa_seq = next(SeqIO.parse(input_AASeq, "fasta")).seq
        nt_seq = next(SeqIO.parse(input_NTSeq, "fasta")).seq
        numeric_aa = self.genetics.parseProtein(aa_seq)
        numeric_nt = self.genetics.parseDNA(nt_seq)
        if not self.verifySequences(numeric_aa, numeric_nt):
            print("Failed...")
            return

        # define arrays for computation
        aa_sequences = np.zeros((num_steps + 1, numeric_aa.shape[0]), dtype=np.int32)
        nt_sequences = np.zeros((num_steps + 1, numeric_nt.shape[0]), dtype=np.int32)
        aa_sequences[0] = numeric_aa
        nt_sequences[0] = numeric_nt
        # run evolutionary trajectory

        for step in range(1, num_steps + 1):
            new_aa, new_nt = self.mutationStep(
                aa_sequences[step - 1], nt_sequences[step - 1], selection_temp
            )
            aa_sequences[step] = new_aa
            nt_sequences[step] = new_nt

        return aa_sequences, nt_sequences

    def writeResultFasta(
        self,
        aa_trajectory: np.array,
        nt_trajectory: np.array,
        aa_filename: str,
        nt_filename: str,
    ) -> None:
        revaa = {v: k for k, v in self.genetics.aa_code.items()}
        revcodon = {v: k for k, v in self.genetics.codon_code.items()}
        aa_seqs = ["".join([revaa[aa] for aa in aa_seq]) for aa_seq in aa_trajectory]
        nt_seqs = ["".join([revcodon[nt] for nt in nt_seq]) for nt_seq in nt_trajectory]

        with open(aa_filename, "w") as out:
            for idx, seq in enumerate(aa_seqs):
                if idx == 0:
                    out.write(">Original Sequence\n")
                    out.write(seq + "\n")
                else:
                    out.write(f">Sequence step {idx}\n")
                    out.write(seq + "\n")

        with open(nt_filename, "w") as out:
            for idx, seq in enumerate(nt_seqs):
                if idx == 0:
                    out.write(">Original Sequence\n")
                    out.write(seq + "\n")
                else:
                    out.write(f">Sequence step {idx}\n")
                    out.write(seq + "\n")

    def compute_Hamiltonian(self, sequences, temperature, interDomainCutoff=None):
        numerical_sequences, headers = create_numerical_MSA(
            sequences, self.dca_params.symboldict
        )
        return (
            return_Hamiltonian(
                numerical_sequences,
                self.dca_params.couplings / temperature,
                self.dca_params.localfields / temperature,
                interDomainCutoff=interDomainCutoff,
            ),
            headers,
        )
