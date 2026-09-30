# WLASL-100 Dataset Metadata

This directory documents the metadata and split configurations for the **WLASL-100** dataset used in the ASL Recognition project.

## Primary Metadata File

Stored in Google Drive during Colab development:
`/content/drive/MyDrive/ASL-Recognition/metadata/wlasl100_metadata_with_split.csv`

## Schema and Column Definitions

| Column Name | Description | Example |
|---|---|---|
| `gloss` | ASL gloss / vocabulary word label | `book` |
| `video_id` | Unique identifier for the video instance | `07069` |
| `signer_id` | Unique identifier for the signer | `1` |
| `original_split` | Split assignment from the original WLASL release | `train` |
| `frame_start` | Starting frame index of the sign segment | `1` |
| `frame_end` | Ending frame index of the sign segment | `-1` |
| `fps` | Video frame rate | `25` |
| `url` | Source video URL | `https://...` |
| `class_id` | Mapped integer class ID (0 to 99) | `0` |
| `our_split` | Validated signer-independent split (`train`, `val`, `test`) | `train` |

## Signer-Independent Split Statistics

- **Total Classes**: 100
- **Total Expected Videos**: 2038
- **Total Signers**: 97

### Partition Summary

- **Train**: 68 signers, 100 classes, 1234 videos
- **Validation**: 15 signers, 100 classes, 426 videos
- **Test**: 14 signers, 100 classes, 378 videos

### Signer Independence Guarantee

- `Train ∩ Validation = ∅`
- `Train ∩ Test = ∅`
- `Validation ∩ Test = ∅`

Zero signer overlap ensures the model generalizes to unseen signers rather than memorizing individual signer characteristics.

## Dataset Availability Note

Currently, approximately ~1,019 videos are available locally/in Drive out of 2038 expected videos. Missing video recovery is underway. Initial pipeline sanity testing uses available videos; final benchmarks require the complete 2038-video set.
