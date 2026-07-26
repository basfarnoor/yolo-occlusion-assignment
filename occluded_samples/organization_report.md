# Organization Report

- Workbook sample rows: **14**
- Expected images (rows x 5 stages): **70**
- Copied: **1**
- Already present (rerun-safe skip): **29**
- Missing: **4**
- Ambiguous: **0**
- Conflicting: **0**
- Empty Excel cells: **36**

## Duplicate sample numbers in the workbook

- Sample number `4` appears in Excel rows [5, 6, 7, 8] (4 occurrences). Each was organized as its own separate folder (by row order) since the underlying filenames differ.
- Sample number `8` appears in Excel rows [12, 13] (2 occurrences). Each was organized as its own separate folder (by row order) since the underlying filenames differ.

## Rows requiring human attention

| Sample no. | Excel row | Stage | Excel value | Status | Notes |
|---|---|---|---|---|---|
| 2 | 3 | previous_no_occlusion | -  | missing | Cell contains placeholder '- ', not a real filename. |
| 5 | 9 | previous_no_occlusion | - | missing | Cell contains placeholder '-', not a real filename. |
| 6 | 10 | previous_no_occlusion | - | missing | Cell contains placeholder '-', not a real filename. |
| 7 | 11 | previous_no_occlusion | -  | missing | Cell contains placeholder '- ', not a real filename. |

## Folder mapping

Folders are numbered by the workbook's row order (`sample_001`, `sample_002`, ...), not by the Excel `Sample no.` value, because that value repeats for rows 4 and 8. The manifest's `sample_number` and `excel_row` columns record the original label and exact row for traceability.

## Notes for the student

- Sample 1's 'Previous No Occlusion' and 'First Partial Occlusion' cells contained the identical filename in the workbook; both stages were copied as that same source image, so those two files in `sample_001/` will look identical. Confirm this was intentional.
- Rows with only a dash placeholder (samples 2, 5, 6, 7) and the fully blank rows (samples 9, 10) produced folders containing zero or very few images; see the manifest.
