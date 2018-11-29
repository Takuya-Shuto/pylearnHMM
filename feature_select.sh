#!/bin/bash
for folder in $(seq 100 2053)
do
    echo $folder
    for subject in $(seq -f %03g 1 54)
    do
        for sequence in $(seq -f %03g 150)
        do
            if [ -e ./data/MMI_custum/Sessions/${folder}/S${subject}-${sequence}.avi ]
            then
                awk -F, 'BEGIN{
                    FS=",";OFS=","
                    } 
                    {
                        print $6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22
                    }'  ./data/MMI_custum/Sessions/${folder}/S${subject}-${sequence}-AU/S${subject}-${sequence}-new.csv > ./data/MMI_custum/Sessions/${folder}/S${subject}-${sequence}-AU/S${subject}-${sequence}-AU.csv
            fi
        done
    done
done