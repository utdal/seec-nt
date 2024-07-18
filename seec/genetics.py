import numpy as np
from scipy.spatial.distance import squareform, pdist
from typing import Union

genetic_code = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "N": ["AAT", "AAC"],
    "D": ["GAT", "GAC"],
    "C": ["TGT", "TGC"],
    "Q": ["CAA", "CAG"],
    "E": ["GAA", "GAG"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "K": ["AAA", "AAG"],
    "M": ["ATG"],
    "F": ["TTT", "TTC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "W": ["TGG"],
    "Y": ["TAT", "TAC"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
}


class GeneticsTools:
    def __init__(self, symboldict: str = "-ACDEFGHIKLMNPQRSTVWY") -> None:
        """Convert nucleotide sequence into integer sequence."""
        self.aa = symboldict
        self.nt = "ACGTU-"
        self.aa_code = {aa: idx for idx, aa in enumerate(self.aa)}
        self.nt_code = {nt: idx for idx, nt in enumerate(self.nt)}

        # Build codon dictionary
        # Simultaneously, build list of all codons and assign them a number.
        # Then make a list of numeric representations of each codon.
        # Use this (numeric_codons) to build adjacency matrix, to use as a
        # lookup table to find which amino acids are one mutation away.
        codons = list()
        numeric_codons = list()
        for array in genetic_code.values():
            for item in array:
                codons.append(item)
                numeric_codons.append([self.nt_code[x] for x in item])
        self.codon_code = {codon: idx for idx, codon in enumerate(codons)}

        # build adjacency matrix, number of mutations needed to go from codon i->j
        self.distance_matrix = (
            squareform(pdist(np.array(numeric_codons), metric="hamming"))
        ) * 3
        self.genetic_code = self.numericGeneticCode()
        self.codonToAA = self.buildCodonToAminoAcid()

    def buildCodonToAminoAcid(self) -> dict:
        codonToAA = dict()
        for item in self.genetic_code.items():
            for value in item[1]:
                codonToAA[int(value)] = item[0]
        return codonToAA

    def numericGeneticCode(self) -> dict:
        output = dict()
        for aa, codonlist in genetic_code.items():
            output[self.aa_code[aa]] = np.array([self.codon_code[c] for c in codonlist])
        return output

    def parseDNA(self, dna_code: str) -> np.array:
        split_code = [dna_code[i : i + 3] for i in range(0, len(dna_code), 3)]
        return np.array(
            [self.codon_code[codon] for codon in split_code], dtype=np.int32
        )

    def parseProtein(self, protein_seq: str) -> np.array:
        return np.array([self.aa_code[aa] for aa in protein_seq], dtype=np.int32)

    def isMutable(self, currentCodon: int, proposalAA: int) -> Union[bool, int]:
        """Check if current codon is one mutation away from producing the proposed amino acid.
        If possible, return a [true, newCodon]. If not, return [false, currentCodon]."""
        distances = self.distance_matrix[currentCodon, self.genetic_code[proposalAA]]
        possibleMutations = distances[distances == 1]  # one nt mutation away allowed
        if len(possibleMutations) == 0:
            return [False, currentCodon]
        else:
            newCodon = np.random.choice(self.genetic_code[proposalAA][distances == 1])
            return [True, newCodon]
