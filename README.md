# Papers Reproducibility

This repository collects paper materials for reproducibility purposes. Each folder contains a complete zipped repository used for a specific research work, allowing anyone to reproduce the results of our published papers.

## Related Libraries

This research builds upon two core libraries:
- [kANNolo](https://github.com/TusKANNy/kannolo) 
- [Seismic](https://github.com/TusKANNy/seismic) 

## Papers

The following papers are included in this repository in chronological order:

- [Multivector Reranking in the Era of Strong First-Stage Retrievers (ECIR 2026)](ecir2026/)
- [Lossless Compression of Proximity Graph Indexes for Efficient Approximate Nearest Neighbors Search (CIKM 2026)](cikm2026/)

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

3. **Unzip the repository archive:**
   ```bash
   unzip *.zip
   ```

4. **Follow the instructions in the unzipped README:**
   Each unzipped folder contains its own `REPRODUCTION_GUIDE.md` with specific instructions to reproduce the results of that particular paper.
