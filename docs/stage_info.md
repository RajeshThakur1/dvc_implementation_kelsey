## Brief information about all stages

### stage_00_template

this stage is just a template file so whenever we have to create new we can use this template to add new stage in this pipeline

### stage_01_load_local_data_in_s3

This stage is responsible for pushing the local data changes in s3
our local data are available in side the data dir of respective project

### stage_02_get_training_data_from_s3
This stage is responsible for pulling the updated data from s3 to our local system

### stage_03_prepare_data
This stage is responsible for preparing the data from the cleaning the data and store in the prepared stage

### stage_04_label_encoder
This stage is responsible for doing the label encoding of intents

### stage_05_training
This stage is responsible for start training



