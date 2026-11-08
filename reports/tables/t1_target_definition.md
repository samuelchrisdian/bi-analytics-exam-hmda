| kode_action_taken   | label_hmda                                    | keterangan                               | y   | perlakuan                            |
|:--------------------|:----------------------------------------------|:-----------------------------------------|:----|:-------------------------------------|
| 1                   | Loan originated                               | Disetujui dan dicairkan                  | 0   | Dipakai                              |
| 2                   | Application approved but not accepted         | Disetujui, tidak diambil pemohon         | 0   | Dipakai                              |
| 3                   | Application denied                            | Ditolak lender                           | 1   | Dipakai                              |
| 4                   | Application withdrawn by applicant            | Ditarik pemohon                          | -   | Dibuang (tidak ada keputusan lender) |
| 5                   | File closed for incompleteness                | Berkas tidak lengkap                     | -   | Dibuang (tidak ada keputusan kredit) |
| 6                   | Purchased loan                                | Kredit dibeli dari lender lain           | -   | Dibuang (bukan aplikasi baru)        |
| 7                   | Preapproval request denied                    | Pra-persetujuan ditolak                  | -   | Dibuang (bukan aplikasi KPR penuh)   |
| 8                   | Preapproval request approved but not accepted | Pra-persetujuan disetujui, tidak diambil | -   | Dibuang (bukan aplikasi KPR penuh)   |
