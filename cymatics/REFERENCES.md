# References and Source Notes

This file records the scientific basis and external-source roles used by the application. URLs are kept here so the software remains inspectable even when the app is used offline.

## DNA structure and coordinate representation

- Calladine CM, Drew HR, Luisi BF, Travers AA. Understanding DNA: the molecule and how it works. Garland/related structural DNA literature.
- Olson WK et al. DNA sequence-dependent structure: base-pair step parameters such as shift, slide, rise, tilt, roll, and twist are standard descriptors in structural DNA modeling.
- 3D coordinate resources such as the Protein Data Bank (PDB) are preferred when an experimentally determined structure is available.

## DNA databases used by the retrieval layer

- ZFIN: https://zfin.org/
- ZFIN hoxb1a gene record: https://zfin.org/ZDB-GENE-990415-101
- ZFIN hoxb1a transcript record: https://zfin.org/ZDB-TSCRIPT-090929-7277
- NCBI Gene: https://www.ncbi.nlm.nih.gov/gene/30337
- Ensembl REST API documentation: https://rest.ensembl.org/documentation/

## Cymatics / resonator physics

- Chladni patterns and mode shapes arise from standing-wave solutions of the resonator; real patterns depend on geometry and boundary conditions.
- Circular membrane modes are Bessel-function solutions with angular and radial mode indices.
- Circular thin plates are described by the plate equation with solutions involving Bessel and modified Bessel functions. Clamped boundaries lead to characteristic equations involving J_m and I_m and their derivatives.

Useful open literature/source pages:

- Circular membrane mode discussion: https://pmc.ncbi.nlm.nih.gov/articles/PMC2711632/
- Chladni/circular plate experimental discussion: https://pmc.ncbi.nlm.nih.gov/articles/PMC10969725/
- Thin circular plate mode theory example: https://pmc.ncbi.nlm.nih.gov/articles/PMC8870825/

## Modeling policy

The application deliberately distinguishes:

1. molecular structure,
2. derived 2-D molecular artwork,
3. mathematical resonator mode fitting,
4. physical resonator calibration,
5. measured experimental image.

Agreement between any two stages should not be treated as validation of the other stages unless the intervening physics has been independently tested.
