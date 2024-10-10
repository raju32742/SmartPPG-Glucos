# 



# dataset requirements

create a dataset as follows:
 
```
dataset_folder
    |-raw_videos
        |-sample_1.mp4
        |-2.mp4
        |-sample-3.mp4
        |-...........
        |-...........

    |-data.csv

```

```
* data.csv colums:

    * SL              : Serial No.
    * Patient's ID    : ID of Subjects (like, 0001, 0901, ...)
    * Name            : Name of subjects (xyz, abcd, ...)
    * Age             : Age of subjects (12,69, ...)
    * Sex(M/F)        : Male or Female (M/F)
    * File_ext(*.mp4) : video file extension (.mp4)
    * Gl (mmol/L)     : Glucose concentration
```

# Execution
- ```conda activate my_env```
- ```cd scripts```
- run: ```./server.sh```
