# Trajectory Data Overview

This dataset contains atomic structures for Li (100) surface incorporating N/O/F from 0 through 3 monolayer (ML).
All structures were optimized by **SCAN functional**.

## Directory Structure

- `trajectory_*/`  
  Each of these folders contains **only the structure at the convex hull minimum**.  
  These files are in ASE-readable `.traj` format.  

- `trajectory_*.zip`  
  Each ZIP file contains **the full structures**.

## Notes

- All trajectory files are compatible with `ase.io.read`.
- Use `ase.io.read("filename.traj", index=-1)` to directly access the SCAN-optimized structure.

