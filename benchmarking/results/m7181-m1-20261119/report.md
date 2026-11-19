# Native versus Slang frontend profiling

Ratios are reported only for matched, correctness-accepted workload hashes.
A ratio above 1 means native was faster. Unavailable metrics remain unavailable.

## Comparative timing and scaling

| Workload | Category | Backend | Native median (s) | Slang median (s) | Native speedup | Structure |
|---|---|---:|---:|---:|---:|---|
| continuous_1024 | semantic | cpu | 0.000346125 | 0.000120833 | 0.3491x | same |
| sequential_1024 | semantic | cpu | 0.000342 | 0.000123375 | 0.3607x | same |
| conditional_10240 | semantic | cpu | 0.00354292 | 0.000743291 | 0.2098x | same |
| arithmetic_10240 | semantic | cpu | 0.00381992 | 0.000666333 | 0.1744x | same |
| sequential_10240 | semantic | cpu | 0.00387925 | 0.000645667 | 0.1664x | same |
| case_10240 | semantic | cpu | 0.00358704 | 0.000700875 | 0.1954x | same |
| boolean_102400 | semantic | cpu | 0.0889108 | 0.00631129 | 0.07098x | same |
| wide_102400 | semantic | cpu | 0.0722879 | 0.00524233 | 0.07252x | same |
| literal_1024 | semantic | cpu | 0.000248042 | 0.000173542 | 0.6996x | same |
| arithmetic_102400 | semantic | cpu | 0.0768354 | 0.00591267 | 0.07695x | same |
| comment_10240 | semantic | cpu | 0.00113592 | 0.000261959 | 0.2306x | same |
| parameter_1024 | semantic | cpu | 0.000254708 | 0.000109416 | 0.4296x | same |
| boolean_1024 | semantic | cpu | 0.000570625 | 0.000397292 | 0.6962x | same |
| continuous_10240 | semantic | cpu | 0.00342192 | 0.000615834 | 0.18x | same |
| sequential_102400 | semantic | cpu | 0.100282 | 0.00540571 | 0.0539x | same |
| identifier_102400 | semantic | cpu | 0.0138555 | 0.00177 | 0.1277x | same |
| parameter_10240 | semantic | cpu | 0.00313533 | 0.000445875 | 0.1422x | same |
| literal_10240 | semantic | cpu | 0.00229821 | 0.00145554 | 0.6333x | same |
| conditional_1024 | semantic | cpu | 0.000345875 | 0.000133292 | 0.3854x | same |
| comment_102400 | semantic | cpu | 0.0145519 | 0.001934 | 0.1329x | same |
| conditional_102400 | semantic | cpu | 0.0743354 | 0.00648617 | 0.08726x | same |
| case_1024 | semantic | cpu | 0.000328917 | 0.000132708 | 0.4035x | same |
| case_102400 | semantic | cpu | 0.0732221 | 0.00606287 | 0.0828x | same |
| comment_1024 | semantic | cpu | 0.000138792 | 8.9375e-05 | 0.6439x | same |
| wide_10240 | semantic | cpu | 0.00346167 | 0.000621292 | 0.1795x | same |
| continuous_102400 | semantic | cpu | 0.0714837 | 0.00521713 | 0.07298x | same |
| identifier_10240 | semantic | cpu | 0.00109113 | 0.00023725 | 0.2174x | same |
| boolean_10240 | semantic | cpu | 0.004188 | 0.000734458 | 0.1754x | same |
| literal_102400 | semantic | cpu | 0.0365147 | 0.0158321 | 0.4336x | same |
| identifier_1024 | semantic | cpu | 0.000143125 | 8.3417e-05 | 0.5828x | same |
| parameter_102400 | semantic | cpu | 0.120896 | 0.00359858 | 0.02977x | same |
| wide_1024 | semantic | cpu | 0.000410333 | 0.000127166 | 0.3099x | same |
| arithmetic_1024 | semantic | cpu | 0.000490292 | 0.000168167 | 0.343x | same |
| continuous_1024 | source_to_netlist | cpu | 0.00049725 | 0.000627458 | 1.262x | different (see JSON) |
| sequential_1024 | source_to_netlist | cpu | 0.000531166 | 0.000487667 | 0.9181x | different (see JSON) |
| conditional_10240 | source_to_netlist | cpu | 0.00869829 | 0.00212725 | 0.2446x | different (see JSON) |
| arithmetic_10240 | source_to_netlist | cpu | 0.0127959 | 0.00213833 | 0.1671x | different (see JSON) |
| sequential_10240 | source_to_netlist | cpu | 0.0100994 | 0.00107746 | 0.1067x | different (see JSON) |
| case_10240 | source_to_netlist | cpu | 0.0108046 | 0.00115179 | 0.1066x | different (see JSON) |
| boolean_102400 | source_to_netlist | cpu | 1.02575 | 0.0209855 | 0.02046x | different (see JSON) |
| wide_102400 | source_to_netlist | cpu | 0.456496 | 0.012659 | 0.02773x | different (see JSON) |
| literal_1024 | source_to_netlist | cpu | 0.000710791 | 0.00083675 | 1.177x | different (see JSON) |
| arithmetic_102400 | source_to_netlist | cpu | 0.702178 | 0.015738 | 0.02241x | different (see JSON) |
| comment_10240 | source_to_netlist | cpu | 0.00196846 | 0.000936125 | 0.4756x | different (see JSON) |
| parameter_1024 | source_to_netlist | cpu | 0.000310125 | 0.000512958 | 1.654x | different (see JSON) |
| boolean_1024 | source_to_netlist | cpu | 0.000993416 | 0.000912334 | 0.9184x | different (see JSON) |
| continuous_10240 | source_to_netlist | cpu | 0.008603 | 0.00178363 | 0.2073x | different (see JSON) |
| sequential_102400 | source_to_netlist | cpu | 0.536324 | 0.00578208 | 0.01078x | different (see JSON) |
| identifier_102400 | source_to_netlist | cpu | 0.0398765 | 0.00401621 | 0.1007x | different (see JSON) |
| parameter_10240 | source_to_netlist | cpu | 0.00426358 | 0.00117788 | 0.2763x | different (see JSON) |
| literal_10240 | source_to_netlist | cpu | 0.00845771 | 0.00264575 | 0.3128x | different (see JSON) |
| conditional_1024 | source_to_netlist | cpu | 0.000502167 | 0.000596917 | 1.189x | different (see JSON) |
| comment_102400 | source_to_netlist | cpu | 0.0625328 | 0.00477963 | 0.07643x | different (see JSON) |
| conditional_102400 | source_to_netlist | cpu | 0.460322 | 0.0160578 | 0.03488x | different (see JSON) |
| case_1024 | source_to_netlist | cpu | 0.000520042 | 0.000546042 | 1.05x | different (see JSON) |
| case_102400 | source_to_netlist | cpu | 0.622253 | 0.00649233 | 0.01043x | different (see JSON) |
| comment_1024 | source_to_netlist | cpu | 0.000178417 | 0.000489459 | 2.743x | different (see JSON) |
| wide_10240 | source_to_netlist | cpu | 0.00863492 | 0.0017675 | 0.2047x | different (see JSON) |
| continuous_102400 | source_to_netlist | cpu | 0.458273 | 0.0125765 | 0.02744x | different (see JSON) |
| identifier_10240 | source_to_netlist | cpu | 0.00162354 | 0.000813 | 0.5008x | different (see JSON) |
| boolean_10240 | source_to_netlist | cpu | 0.0167042 | 0.00265512 | 0.1589x | different (see JSON) |
| literal_102400 | source_to_netlist | cpu | 0.222667 | 0.0233263 | 0.1048x | different (see JSON) |
| identifier_1024 | source_to_netlist | cpu | 0.000183291 | 0.00045875 | 2.503x | different (see JSON) |
| parameter_102400 | source_to_netlist | cpu | 0.186115 | 0.00702875 | 0.03777x | different (see JSON) |
| wide_1024 | source_to_netlist | cpu | 0.000563042 | 0.00065425 | 1.162x | different (see JSON) |
| arithmetic_1024 | source_to_netlist | cpu | 0.000874417 | 0.000777916 | 0.8896x | different (see JSON) |
| continuous_1024 | syntax | cpu | 0.000172541 | 8.0625e-05 | 0.4673x | same |
| sequential_1024 | syntax | cpu | 0.000166167 | 8.1875e-05 | 0.4927x | same |
| conditional_10240 | syntax | cpu | 0.001901 | 0.000369875 | 0.1946x | same |
| arithmetic_10240 | syntax | cpu | 0.00189629 | 0.000362375 | 0.1911x | same |
| sequential_10240 | syntax | cpu | 0.00168721 | 0.000332334 | 0.197x | same |
| case_10240 | syntax | cpu | 0.00191929 | 0.000390292 | 0.2034x | same |
| boolean_102400 | syntax | cpu | 0.0209977 | 0.00325017 | 0.1548x | same |
| wide_102400 | syntax | cpu | 0.0158483 | 0.00261162 | 0.1648x | same |
| literal_1024 | syntax | cpu | 0.000121334 | 8.2333e-05 | 0.6786x | same |
| arithmetic_102400 | syntax | cpu | 0.0178953 | 0.00290646 | 0.1624x | same |
| comment_10240 | syntax | cpu | 0.000592583 | 0.000156666 | 0.2644x | same |
| parameter_1024 | syntax | cpu | 0.000121834 | 7.7458e-05 | 0.6358x | same |
| boolean_1024 | syntax | cpu | 0.000364667 | 0.000120291 | 0.3299x | same |
| continuous_10240 | syntax | cpu | 0.00161208 | 0.000324333 | 0.2012x | same |
| sequential_102400 | syntax | cpu | 0.0164464 | 0.00265408 | 0.1614x | same |
| identifier_102400 | syntax | cpu | 0.00613775 | 0.000983041 | 0.1602x | same |
| parameter_10240 | syntax | cpu | 0.00107754 | 0.000256167 | 0.2377x | same |
| literal_10240 | syntax | cpu | 0.00113817 | 0.00030975 | 0.2721x | same |
| conditional_1024 | syntax | cpu | 0.000192166 | 8.5958e-05 | 0.4473x | same |
| comment_102400 | syntax | cpu | 0.00562167 | 0.00104104 | 0.1852x | same |
| conditional_102400 | syntax | cpu | 0.0177745 | 0.00295938 | 0.1665x | same |
| case_1024 | syntax | cpu | 0.000190125 | 8.7167e-05 | 0.4585x | same |
| case_102400 | syntax | cpu | 0.0181346 | 0.00320954 | 0.177x | same |
| comment_1024 | syntax | cpu | 6.8e-05 | 6.4958e-05 | 0.9553x | same |
| wide_10240 | syntax | cpu | 0.00163033 | 0.0003315 | 0.2033x | same |
| continuous_102400 | syntax | cpu | 0.0158157 | 0.00260692 | 0.1648x | same |
| identifier_10240 | syntax | cpu | 0.0006365 | 0.000150959 | 0.2372x | same |
| boolean_10240 | syntax | cpu | 0.00226804 | 0.000399791 | 0.1763x | same |
| literal_102400 | syntax | cpu | 0.011584 | 0.00246408 | 0.2127x | same |
| identifier_1024 | syntax | cpu | 8.375e-05 | 6.4e-05 | 0.7642x | same |
| parameter_102400 | syntax | cpu | 0.0105519 | 0.00195183 | 0.185x | same |
| wide_1024 | syntax | cpu | 0.000206875 | 8.7625e-05 | 0.4236x | same |
| arithmetic_1024 | syntax | cpu | 0.000263416 | 0.000110541 | 0.4196x | same |

## Ranked measured phase contributions

No independently measured phase timings were available; no hotspot ranking is claimed.
