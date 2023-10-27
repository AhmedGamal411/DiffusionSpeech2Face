[COMMON]
datasetPathVideo = Path to video files. The videos can be in inner folders.
datasetPathDatabase = Path to create the database and read from it later
cuda= Whether to use CUDA. Must be 1
cpus= No. of CPU cores to use then multithreading or multiprocessing. A good no. to use is half the no. of CPUs in your machine.
resizeImageTo = Resize image to a certain width x height. This is also used as the output dims of the network. Use a lower number if OOM errors occur.
audio_embs_options = wav2vec or openl3 or pyannoteTitaNet
audio_embs = wav2vec or openl3 are depreacated. Must use pyannoteTitaNet
unet1_dim = 64
unet1_image_size = 16
insert_amd_env_vars = Input 0 to not insert AMD environment variables. Anything else to insert them. The environment variables are HSA_OVERRIDE_GFX_VERSION and ROCM_PATH. They are sometimes used to get CUDA working on AMD machines after installing it.
HSA_OVERRIDE_GFX_VERSION = Get CUDA working on newer AMD than those supported by overriding GFX version
ROCM_PATH = ROCm Path

[dbCreateAndPopulate]
recreateDb= Insert 0 to insert into a previously created database, or insert anything else to recreate the database from scratch. 

[extractAudio]
datasetPathAudio = Path to extract audio files. Audio files will be deleted so will be empty.
dbChunk = Chunk of video files to get from database at a time.
time_to_wait_before_deleting_files = Time to wait before deleting files in seconds. If the program is stopped abruptly please delete any audio files created in the execution folder.


[extractFaces]
expandFaceVerticalRatio = Expand vertical cropped face image by a ratio of the cropped face image height. Do not change.
expandFaceHorizontalRatio= Expand horizontal cropped face image by a ratio of the cropped face image width. Do not change.
faceDetectionDeepFaceBackend= Which backend to use for face detection and cropping.  'opencv' or 'ssd' or 'dlib' or 'mtcnn' or 'retinaface' or 'mediapipe'. Do not change.
parallelism = Chunk of video files to get from database at a time.
parallelismFrames = No of frames to extract per video. Note: Regardless of no. of frames extracted, only one will be used in training. So better choose 1.
datasetPathFrames = Path to export frames. Frames will be deleted so will be empty.
datasetPathFaces = Path to export face images.

[fineTuneStableDiffusionTraining] DEPREACATED
db_chunk = 50000
dev_mode = 0
continue_from_epoch = 1
continue_from_offset = 0
continue_from_epoch_and_offset = 0
unconditional_guidance_scale = 40

[extractOpenL3] DEPREACATED
datasetPathAudio = /home/gamal/Datasets/Dataset1/Audio
dbChunk = 240
time_to_wait_before_deleting_files = 60
openl3_mode_options = stable or imagen
openl3_mode = imagen

[extractWavToVec] DEPREACATED
datasetPathAudio = /home/gamal/Datasets/Dataset1/Audio
dbChunk = 240
time_to_wait_before_deleting_files = 60
audio_length_wav2vec = 6

[extractPyannoteTitaNet]
datasetPathAudio = Path to extract audio files. Audio files will be deleted so will be empty.
dbChunk = Chunk of video files to get from database at a time.
time_to_wait_before_deleting_files = Time to wait before deleting files in seconds. If the program is stopped abruptly please delete any audio files created in the execution folder.

[extractAudioFeatures]
datasetPathAudio = /home/gamal/Datasets/Dataset1/Audio
dbChunk = 240
time_to_wait_before_deleting_files = 60

[fineTuneStableDiffusionTesting] DEPREACATED
use_video_in_configuration = 1
video_path = /home/gamal/Datasets/Dataset1/Video/blhA_I4zjvE_15.600000_22.333333.mp4
time_to_wait_before_deleting_files = 180
dev_mode = 0

[train_imagen]
model_filename = Model filename. The actual filename will have _(audio_length_used)s added to it. train_imagen_all file will use this file or create one if it is not created.
audio_length_used = 6 or 12 or 24
epochs = No. of epochs to train
sample_every_offset = 2 depreacated
save_every_offset = 2 depreacated
sub_epochs = No. of training steps for a single batch
sample_every = Get a test image using one of the inputs every this no. of training steps.
save_model_every = Save the model every this no. of training steps.
batch_size = Batch size. Note that gradient accumulation is used and effective batch size is batch_size x 8. Decrease if you run into OOM errors.
continue_from_epoch = If continue_from_epoch_and_offset_flag is any number but zero. Continue from this epoch.
continue_from_offset = If continue_from_epoch_and_offset_flag is any number but zero. Continue from this database offset.
continue_from_epoch_and_offset_flag = If any number but zero, continue from a certain epoch and offset. Used to continue training from a save point.
db_chunk = Data chunk to get from database at a time. Reduce if memory isn't sufficient.

[test_imagen]
video_path = v.mp4
time_to_wait_before_deleting_files = 180
audio_length_used = 24
model_filename = model_live
openl3_mode_options = stable or imagen
openl3_mode = imagen
folder = imagen-tests-folder
number_of_images = 3
age = 25
ethnicity = black 
gender = man
language = English


[extractVggBlurred]
dbChunk = 20
boxBlurMin = 4
boxBlurMax = 14
gaussianBlurMin=2
gaussianBlurMax=6


python dbCreateAndPopulate.py
python dbCreateAdditional.py
python dbCreateFacesBlurred.py
python extractAudio.py
python extractFaces.py

until python extractPyannoteTitaNet.py
do
    echo "Restarting"
    sleep 2
done

until python extractAudioFeatures.py
do
    echo "Restarting"
    sleep 2
done

until python extractVggBlurred.py
do
    echo "Restarting"
    sleep 2
done

until python extractAudioTransformer.py
do
    echo "Restarting"
    sleep 2
done

train_imagen_all_u1.ipynb
train_imagen_all_u2.ipynb




    pip uninstall protobuf

    pip install protobuf==3.9.2
    pip install protobuf


    find . -maxdepth 1 -type f -print0 | head -z -n 10000 | xargs -0 -r -- cp -t "/media/gamal/Passport/Datasets/VoxCeleb2/Voxceleb2VQVAETrain" --
