# Data

Put licensed, manually obtained datasets under this directory (ignored by Git). For GenImage, use the [official repository](https://github.com/GenImage-Dataset/GenImage), follow its Google Drive or Baidu archive links, and preserve `<generator>/{train,val}/{ai,nature}/...`; Baidu's posted access code is `ztf1`. Use `scripts/prepare_genimage.py`. Read and accept GenImage's CC BY-NC-SA 4.0 plus additional noncommercial research terms before downloading/using it. No data downloader runs automatically.

Primary/independent sources are documented in `../docs/dataset_protocol.md`. Store generated manifests here; they contain paths and fingerprints, not image content. Record seed, source metadata, and exclusions.
