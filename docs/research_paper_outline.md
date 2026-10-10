# Research Paper Outline: De Novo Generation of DRD2 Inhibitors

**Target Venue:** IEEE International Conference on Bioinformatics and Biomedicine (BIBM) or ACM Bioinformatics
**Format:** IEEE standard double-column (6-8 pages max)

---

## 1. Abstract (approx. 200 words)
*   **The Problem:** Schizophrenia treatment relies on DRD2 antagonists, but current clinical drugs (like Haloperidol or Risperidone) suffer from severe side effects and lack of novelty.
*   **The Method:** We implemented a Deep Reinforcement Learning (RL) framework utilizing Proximal Policy Optimization (PPO). The agent generates novel SMILES strings and is rewarded dynamically using a physics-in-the-loop AutoDock Vina scoring function.
*   **The Result:** The model successfully generated a Novel Chemical Entity (NCE) that achieves a binding affinity of -9.53 kcal/mol, outperforming 5 major clinical baseline drugs, while maintaining high Blood-Brain Barrier permeability and zero predicted toxicity.

## 2. Introduction
*   Brief overview of the DRD2 receptor and its role in CNS disorders.
*   The high failure rate and cost of traditional High-Throughput Screening (HTS).
*   The rise of Generative AI in drug discovery.
*   **Our Contribution:** We demonstrate an end-to-end RL pipeline that combines transformer-based language models (MolGPT) with physics-based reward functions to design highly optimized, novel scaffolds.

## 3. Related Work
*   Previous attempts using Variational Autoencoders (VAEs) and GANs for molecule generation.
*   Existing RL approaches (e.g., REINVENT), highlighting their limitations (often relying on simple 2D structural rewards rather than 3D physics).
*   How our approach bridges the gap between deep learning and thermodynamic simulation.

## 4. Methodology (The Core Computer Science Section)
*   **4.1 Generative Model (Policy Network):** Explain the MolGPT transformer architecture and how it learns the grammar of SMILES strings.
*   **4.2 Reinforcement Learning Framework (PPO):** Detail how PPO was used. Explain the state (current SMILES string), action (next character), and policy updates.
*   **4.3 Physics-in-the-Loop Reward Function:** This is the star of the paper! Explain how AutoDock Vina was automated to evaluate the 3D binding affinity inside the RL loop.
*   **4.4 Hardware & Batch Optimization:** Briefly mention how SMILES generation and 3D conversion were batched to run efficiently on an RTX 3090.

## 5. Experimental Setup
*   **Target Receptor:** DRD2 (PDB ID: 6LUQ) and the Asp114 binding pocket.
*   **Baselines:** Explain why we selected Haloperidol, Risperidone, Clozapine, Aripiprazole, and Olanzapine as the clinical benchmarks.
*   **Evaluation Metrics:** Vina Binding Energy (kcal/mol), Tanimoto Similarity (for novelty), ADMET profiling (SwissADME/ProTox-II).

## 6. Results & Discussion
*   **6.1 RL Optimization Trajectory:** (Insert a graph showing the reward/docking score improving over thousands of epochs).
*   **6.2 Comparative Docking Performance:** (Insert the beautiful bar chart we generated today showing Molecule 1 beating the 5 clinical drugs).
*   **6.3 ADMET and Biological Safety:** Present the tables proving BBB permeability and Class 4 toxicity.
*   **6.4 Novelty Certification:** Discuss the PubChem API results and Tanimoto similarity to prove the AI didn't just copy a known drug.

## 7. Conclusion & Future Work
*   Summarize that the AI pipeline successfully engineered a computationally validated drug candidate.
*   **Future Work:** Mention that the next phase involves 50ns Molecular Dynamics (GROMACS) simulations and eventual wet-lab *in-vitro* synthesis.

## 8. References
*   Cite AutoDock Vina, RDKit, PPO algorithms, original MolGPT paper, SwissADME, etc.
