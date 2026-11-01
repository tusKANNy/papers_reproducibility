# Papers Reproducibility

This repository collects paper materials for reproducibility purposes. Each folder contains the complete repository used for a specific research work, either zipped or as plain files, allowing anyone to reproduce the results of our published papers.

## Related Libraries

This research builds upon three core libraries:
- [kANNolo](https://github.com/TusKANNy/kannolo) 
- [Seismic](https://github.com/TusKANNy/seismic) 
- [Vectorium](https://github.com/TusKANNy/vectorium) 

## Papers

The following papers are included in this repository in chronological order:

- [Multivector Reranking in the Era of Strong First-Stage Retrievers (ECIR 2026)](ecir2026/)
- [Lossless Compression of Proximity Graph Indexes for Efficient Approximate Nearest Neighbors Search (CIKM 2026)](cikm2026/)
- [Modular and Efficient Approximate Nearest Neighbors Search with Vectorium and kANNolo (Information Systems, submitted)](information-systems-vectorium-kannolo/)

## How to Reproduce Results

To reproduce the results of a specific work, follow these steps:

1. **Clone this repository:**
   ```bash
   git clone https://github.com/TusKANNy/papers_reproducibility.git
   cd papers_reproducibility
   ```

2. **Navigate to the relevant paper folder:**
   ```bash
   cd <paper_folder>
   ```
   For example, for the ECIR 2026 paper:
   ```bash
   cd ecir2026
   ```

3. **Unzip the repository archive, if the folder contains one:**
   ```bash
   unzip *.zip
   ```

4. **Follow the instructions in the unzipped README:**
   Each unzipped folder contains its own `REPRODUCTION_GUIDE.md` with specific instructions to reproduce the results of that particular paper. Folders that are not zipped contain these instructions in their `README.md`.
