# Architecture

The repository separates dataset tooling (`src/imad/data`), experiment scripts, and future API/UI layers. Current image ingestion is offline and manifest-driven. The initial training script is transfer learning with torchvision ResNet-18; model scores are not currently exposed through an API. Backend and frontend remain to be implemented.
