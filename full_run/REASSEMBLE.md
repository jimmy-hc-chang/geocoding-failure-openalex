# Reassembling `affil_full.csv.gz`

`affil_full.csv.gz` (11.9 MB) exceeds GitHub's web-upload size limit, so it is
deposited as three sequential parts. Reassemble before running any script:

**Linux / macOS**

    cat affil_full.csv.gz.part-* > affil_full.csv.gz
    md5sum -c affil_full.csv.gz.md5      # expect: OK

**Windows (PowerShell)**

    cmd /c copy /b affil_full.csv.gz.part-aa+affil_full.csv.gz.part-ab+affil_full.csv.gz.part-ac affil_full.csv.gz
    Get-FileHash affil_full.csv.gz -Algorithm MD5   # expect: BB46B3760CC773A272002DCBCC1DDD3F

Then decompress as usual (`gunzip -k affil_full.csv.gz`, or any archive tool).

Expected result: 319,968 rows, columns
`Paper_ID, Title, Year, Cited_By_Count, Author_ID, Author_Name,
Raw_Affiliation, Standardized_Institutions, Institution_Countries`.
