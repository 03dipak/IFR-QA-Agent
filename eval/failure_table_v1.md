# v1 failure table

rows=236 (228 team + 8 preserved), accuracy=92/236 (39.0%), abstain_ok=157/236, mean_tokens=1717.5

gold_error=56, agent_bug=88

| id | classification | got | expected | abstain_ok |
|---|---|---|---|---|
| G-002 | agent_bug | ['12626.2', '5424.66', '7201.54'] | ['21557.12', '7717.67'] | True |
| G-005 | agent_bug | ['91178', '962399'] | ['1319718', '202411'] | False |
| G-007 | agent_bug | ['16232.28'] | ['16232.28', '6'] | True |
| G-008 | agent_bug | ['11210.26', '5043.57', '6166.69'] | ['16232.28', '6166.69'] | True |
| G-009 | agent_bug | ['211.57', '5022.02'] | ['5022.02', '7'] | False |
| G-011 | agent_bug | ['12804.7', '326469.3', '339274'] | ['494266', '497849.7'] | True |
| G-018 | agent_bug | ['12929.41', '8680.68'] | ['102603053', '8680.68'] | False |
| G-019 | gold_error | ['17479.91'] | ['1', '17479.91'] | True |
| G-020 | agent_bug | ['11802.75', '4872.85', '6929.9'] | ['17479.91', '6929.9'] | True |
| G-023 | gold_error | ['65406.69'] | [] | True |
| G-026 | gold_error | ['4081'] | [] | False |
| G-028 | gold_error | [] | [] | True |
| G-031 | gold_error | ['91790'] | [] | False |
| G-036 | agent_bug | [] | ['1755986'] | False |
| G-037 | agent_bug | [] | ['515096'] | False |
| G-038 | agent_bug | [] | ['830198'] | False |
| G-045 | agent_bug | [] | ['6'] | True |
| G-047 | agent_bug | ['211.57'] | ['7'] | False |
| G-048 | agent_bug | ['9053'] | ['9053'] | False |
| G-049 | agent_bug | ['802871'] | ['1634439'] | True |
| G-050 | agent_bug | ['2160'] | ['2160'] | False |
| G-056 | agent_bug | [] | ['104727'] | False |
| G-057 | agent_bug | ['79020'] | ['42042'] | True |
| G-059 | agent_bug | [] | ['826189'] | False |
| G-060 | agent_bug | [] | ['930529'] | False |
| G-061 | gold_error | ['604'] | ['654'] | True |
| G-062 | agent_bug | ['7792'] | ['24'] | True |
| G-065 | agent_bug | [] | ['90.09'] | False |
| G-067 | agent_bug | [] | ['9'] | False |
| G-068 | agent_bug | [] | ['27909.75'] | False |
| G-070 | agent_bug | ['23429.4'] | ['24029.2'] | False |
| G-071 | agent_bug | [] | ['2024', '2026', '402000', '604848'] | False |
| G-072 | agent_bug | [] | ['2016939', '2024', '2025', '2365801'] | False |
| G-073 | agent_bug | ['2116.3'] | ['2024', '2026', '2116.3', '220.33'] | False |
| G-074 | agent_bug | ['34589.83'] | ['2024', '2026', '24439.05', '34589.83'] | False |
| G-075 | agent_bug | ['18866'] | ['17901', '2024', '2025', '29'] | False |
| G-076 | agent_bug | ['14294'] | ['12978', '14294', '2024', '2025'] | False |
| G-077 | agent_bug | ['0', '47'] | ['147557', '2024', '2025', '47'] | True |
| G-078 | agent_bug | ['164699'] | ['156212', '164699', '2024', '2025'] | False |
| G-079 | agent_bug | ['13136.14'] | ['2025', '2026', '29'] | False |
| G-080 | agent_bug | ['9731.94'] | ['2025', '2026', '8838.55', '9731.94'] | False |
| G-081 | agent_bug | [] | ['2025', '2026', '8838.55', '9731.94'] | False |
| G-082 | agent_bug | [] | ['2024', '2026', '6'] | False |
| G-083 | agent_bug | ['0', '2513.47'] | ['10', '2024', '2026'] | True |
| G-084 | agent_bug | [] | ['2024', '2026', '25774.09', '28162.59'] | False |
| G-085 | agent_bug | ['32717.82'] | ['2024', '2026', '32717.82', '42495.98'] | False |
| G-087 | agent_bug | [] | ['206887.8', '223707.6'] | False |
| G-088 | agent_bug | ['30'] | ['211739.7', '29'] | False |
| G-089 | agent_bug | ['3.29'] | ['0.1', '3.29'] | True |
| G-094 | agent_bug | [] | ['137693.65', '245634.06'] | False |
| G-095 | agent_bug | [] | ['144110.34', '24'] | False |
| G-102 | agent_bug | [] | ['26528.82', '29440'] | False |
| G-103 | agent_bug | [] | ['26528.82', '27.81'] | False |
| G-104 | agent_bug | [] | ['20034', '27.81'] | False |
| G-110 | agent_bug | [] | ['20034', '27.81'] | False |
| G-112 | agent_bug | [] | ['26528.82', '29440'] | False |
| G-113 | agent_bug | [] | ['26528.82', '27.81'] | False |
| G-115 | agent_bug | [] | ['12143.19', '20034'] | False |
| G-119 | agent_bug | [] | ['26528.82', '27.81'] | False |
| G-123 | agent_bug | ['29474'] | ['29440'] | True |
| G-124 | gold_error | ['26750'] | [] | True |
| G-127 | agent_bug | ['16714.19'] | ['27.81'] | False |
| G-128 | gold_error | [] | [] | False |
| G-132 | gold_error | [] | [] | True |
| G-137 | gold_error | [] | [] | True |
| G-138 | gold_error | [] | [] | True |
| G-139 | gold_error | [] | [] | True |
| G-147 | gold_error | [] | [] | True |
| G-148 | gold_error | [] | [] | True |
| G-149 | gold_error | [] | [] | True |
| G-150 | gold_error | [] | [] | True |
| G-151 | gold_error | [] | [] | True |
| G-152 | gold_error | [] | [] | False |
| G-153 | gold_error | [] | [] | True |
| G-154 | gold_error | [] | [] | True |
| G-157 | gold_error | [] | [] | True |
| G-158 | gold_error | [] | [] | True |
| G-160 | gold_error | [] | [] | True |
| G-161 | gold_error | [] | [] | True |
| G-163 | gold_error | [] | [] | True |
| G-164 | gold_error | [] | [] | True |
| G-166 | gold_error | [] | [] | True |
| G-167 | gold_error | [] | [] | True |
| G-168 | gold_error | [] | [] | True |
| G-171 | gold_error | ['1159', '90631', '91790'] | [] | False |
| G-172 | gold_error | [] | [] | True |
| G-173 | gold_error | [] | [] | True |
| G-174 | gold_error | ['13969.7', '18953.88'] | [] | True |
| G-175 | gold_error | [] | [] | True |
| G-176 | gold_error | [] | [] | False |
| G-177 | gold_error | [] | [] | True |
| G-178 | gold_error | [] | [] | True |
| G-179 | gold_error | [] | [] | True |
| G-180 | gold_error | [] | [] | True |
| G-181 | gold_error | [] | [] | True |
| G-182 | gold_error | ['1044.31', '14724.04', '629011', '65322100000', '91186', '920401', '9476.29'] | [] | True |
| G-183 | gold_error | [] | [] | True |
| G-185 | gold_error | [] | [] | True |
| G-186 | gold_error | [] | [] | True |
| G-187 | gold_error | ['20421.97'] | [] | False |
| G-188 | gold_error | [] | [] | True |
| G-190 | gold_error | ['25'] | [] | True |
| G-191 | gold_error | ['16714.19'] | [] | True |
| G-194 | gold_error | [] | [] | True |
| G-195 | gold_error | ['27'] | [] | False |
| G-196 | gold_error | [] | [] | True |
| G-198 | gold_error | ['2277.73'] | [] | True |
| G-200 | agent_bug | [] | ['10080'] | False |
| G-201 | agent_bug | [] | ['6387'] | False |
| G-202 | agent_bug | [] | ['13391', '29440'] | False |
| G-203 | agent_bug | [] | ['13391', '21018.15'] | False |
| G-204 | agent_bug | [] | ['13391', '2361.75'] | False |
| G-205 | agent_bug | [] | ['13391', '29440'] | False |
| G-206 | agent_bug | [] | ['13391', '21018.15'] | False |
| P-001 | agent_bug | ['19'] | ['18898800000'] | True |
| P-002 | agent_bug | ['9793.77'] | ['1122500000'] | True |
| P-003 | agent_bug | [] | ['8.23'] | False |
| P-004 | agent_bug | [] | ['47890000000'] | False |
| P-005 | gold_error | ['604'] | ['340000000'] | True |
| P-006 | agent_bug | ['19', '9793.77'] | ['1122500000', '17776300000', '18898800000'] | True |
| QT-001 | agent_bug | ['17084.8'] | ['78684500000'] | False |
| QT-002 | gold_error | ['17084.8'] | ['1', '2026', '735629', '78684500000'] | False |
| QT-003 | agent_bug | [] | ['14681700000'] | False |
| QT-005 | agent_bug | [] | ['28481700000'] | False |
| QT-006 | agent_bug | [] | ['1', '2026', '278972', '28481700000'] | False |
| QT-007 | agent_bug | ['103838'] | ['7500900000'] | False |
| QT-009 | agent_bug | [] | ['28626200000'] | False |
| QT-010 | agent_bug | [] | ['1', '2026', '245847', '28626200000'] | False |
| QT-011 | agent_bug | ['905.42'] | ['4326000000'] | False |
| QT-013 | agent_bug | [] | ['44168300000'] | False |
| QT-014 | agent_bug | [] | ['1', '2026', '44168300000', '576365'] | False |
| QT-015 | agent_bug | [] | ['12892400000'] | False |
| QT-017 | agent_bug | [] | ['54869100000'] | False |
| QT-018 | agent_bug | [] | ['1', '2026', '54869100000', '751180'] | False |
| QT-019 | agent_bug | [] | ['-35064600000'] | False |
| QT-021 | agent_bug | ['22426'] | ['5371000000'] | True |
| QT-022 | gold_error | ['169270'] | ['1', '2026', '5371000000', '58330'] | True |
| QT-023 | agent_bug | ['22426'] | ['-1067000000'] | True |
| QT-025 | agent_bug | [] | ['47886000000'] | False |
| QT-026 | agent_bug | ['49106000000'] | ['1', '2026', '47886000000', '479480'] | True |
| QT-027 | agent_bug | [] | ['-1220000000'] | False |
| QT-029 | agent_bug | [] | ['44527400000'] | False |
| QT-030 | gold_error | ['10986.68'] | ['1', '2026', '44527400000', '466451'] | False |
| QT-031 | agent_bug | [] | ['1702300000'] | False |
